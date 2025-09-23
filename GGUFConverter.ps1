function Get-EnhancedConverterSource {
    return @'
/*
  AdvancedGGUF_Converter v6.0 — GPU-first (nvCOMP batched ZSTD) + multi-stream GPU SHA256 + official GGUF
  - Drop-in single-file C++ source. Compile with nvcc or host compiler that links CUDA/nvCOMP libs.
  - Requires: CUDA toolkit (runtime), nvCOMP (batched API), gguf/ggml C headers & libs, zstd (fallback), nlohmann::json, OpenSSL (fallback).
  - Example build (linux):
      nvcc -O3 -std=c++17 -Xcompiler -fPIC converter.cpp -o converter \
           -I/path/to/json -I/path/to/gguf/include -I/usr/local/cuda/include -I/path/to/nvcomp/include \
           -L/path/to/libs -lnvcomp -lcudart -lgguf -lzstd -lssl -lcrypto
*/

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cstdarg>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <exception>
#include <memory>
#include <queue>
#include <condition_variable>
#include <sstream>
#include <iomanip>
#include <optional>

#include "json.hpp"
using json = nlohmann::json;

#define GGML_BUILD
#include "ggml.h"
#include "gguf.h"

#include <cuda_runtime.h>
#include <nvcomp.h>   // batched nvCOMP C API
#include <zstd.h>
#include <openssl/sha.h>

// DFloat11 extern API (optional - for CPU fallback)
extern "C" {
    size_t DFloat11_compress_bound(size_t size);
    int    DFloat11_compress(const uint8_t* src, size_t src_size, uint8_t* dst, size_t* dst_size);
    int    DFloat11_decompress(const uint8_t* src, size_t src_size, uint8_t* dst, size_t dst_size);
}

// ---------------------------
// Configuration
// ---------------------------
static constexpr size_t CHUNK_MIN = 4ULL * 1024 * 1024;        // 4 MiB
static constexpr size_t CHUNK_DEFAULT = 16ULL * 1024 * 1024;  // 16 MiB
static constexpr size_t CHUNK_MAX = 128ULL * 1024 * 1024;     // 128 MiB (depending on GPU memory)
static constexpr const char* CHECKPOINT_FILENAME = "convert_checkpoint.json";
static constexpr const char* TELEMETRY_CSV = "convert_telemetry.csv";
static constexpr const char* TELEMETRY_JSON = "convert_telemetry.json";

// ---------------------------
// Logging
// ---------------------------
enum LogLevel { DEBUG=0, INFO=1, WARN=2, ERROR=3 };
static std::mutex g_log_mutex;
static void log_message(LogLevel level, const char* fmt, ...) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    const char* names[] = {"DEBUG","INFO","WARN","ERROR"};
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    printf("[%s.%03d] [%s] ", buf, (int)ms.count(), names[level]);
    va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
    printf("\n"); fflush(stdout);
}
#define LOGD(...) log_message(DEBUG, __VA_ARGS__)
#define LOGI(...) log_message(INFO,  __VA_ARGS__)
#define LOGW(...) log_message(WARN,  __VA_ARGS__)
#define LOGE(...) log_message(ERROR, __VA_ARGS__)

// ---------------------------
// Cancellation / signals
// ---------------------------
static std::atomic<bool> g_cancelled{false};
#ifdef _WIN32
#include <windows.h>
BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT) {
        LOGI("Cancellation requested by user");
        g_cancelled = true;
        return TRUE;
    }
    return FALSE;
}
#else
#include <signal.h>
void signal_handler(int s) {
    LOGI("Signal %d received -> cancellation", s);
    g_cancelled = true;
}
#endif

// ---------------------------
// Utility: CPU SHA256 (fallback or verification)
static std::string sha256_hex_cpu(const uint8_t* data, size_t len) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(data, len, hash);
    std::ostringstream ss; ss << std::hex << std::setfill('0');
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) ss << std::setw(2) << (int)hash[i];
    return ss.str();
}

// ---------------------------
// Highly-optimized multi-stream GPU SHA-256 (tree-hash)
// - per-segment parallel SHA256 kernel (one block per segment)
// - pairwise reduction kernel to hash concatenated digests
// - final root digest returned as hex string
// Notes: This tree mode is high-throughput and secure; it differs from streaming SHA256 (single linear digest).
// Use tree mode for fast integrity; if canonical single-block SHA256 is required, adapt or use CPU fallback.
// ---------------------------

__device__ __constant__ uint32_t d_sha256_k[64] = {
  0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
  0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
  0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
  0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
  0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
  0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
  0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
  0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

__device__ inline uint32_t rotr32(uint32_t x, int r) { return (x >> r) | (x << (32 - r)); }

// Per-segment SHA256 kernel (one block per segment)
extern "C" __global__
void sha256_segment_kernel(const uint8_t* __restrict__ dev_buf, size_t total_len,
                           size_t segment_bytes, uint8_t* __restrict__ dev_out_hashes, size_t out_stride) {
    size_t seg_idx = blockIdx.x;
    size_t seg_offset = seg_idx * segment_bytes;
    if (seg_offset >= total_len) return;
    size_t remaining = total_len - seg_offset;
    size_t this_len = remaining < segment_bytes ? remaining : segment_bytes;
    const uint8_t* seg_ptr = dev_buf + seg_offset;

    extern __shared__ uint32_t s_mem[]; // dynamic shared for W + state
    uint32_t* W = s_mem;                // 64 words
    uint32_t* s_state = s_mem + 64;     // 8 words

    if (threadIdx.x == 0) {
        s_state[0] = 0x6a09e667u;
        s_state[1] = 0xbb67ae85u;
        s_state[2] = 0x3c6ef372u;
        s_state[3] = 0xa54ff53au;
        s_state[4] = 0x510e527fu;
        s_state[5] = 0x9b05688cu;
        s_state[6] = 0x1f83d9abu;
        s_state[7] = 0x5be0cd19u;
    }
    __syncthreads();

    size_t num_blocks = (this_len + 8 + 63) / 64;
    for (size_t blk = 0; blk < num_blocks; ++blk) {
        // W[0..15]
        for (int i = threadIdx.x; i < 16; i += blockDim.x) {
            size_t byte_index = blk * 64 + i * 4;
            uint32_t val = 0;
            for (int b = 0; b < 4; ++b) {
                size_t idx = byte_index + b;
                uint8_t byte = 0;
                if (idx < this_len) byte = seg_ptr[idx];
                else if (idx == this_len) byte = 0x80;
                val = (val << 8) | byte;
            }
            W[i] = val;
        }
        __syncthreads();

        if (blk == num_blocks - 1) {
            uint64_t bits_len = (uint64_t)this_len * 8;
            if (threadIdx.x == 0) {
                W[14] = (uint32_t)((bits_len >> 32) & 0xFFFFFFFFu);
                W[15] = (uint32_t)(bits_len & 0xFFFFFFFFu);
            }
        }
        __syncthreads();

        for (int t = 16 + threadIdx.x; t < 64; t += blockDim.x) {
            uint32_t s0 = rotr32(W[t-15],7) ^ rotr32(W[t-15],18) ^ (W[t-15] >> 3);
            uint32_t s1 = rotr32(W[t-2],17) ^ rotr32(W[t-2],19) ^ (W[t-2] >> 10);
            W[t] = W[t-16] + s0 + W[t-7] + s1;
        }
        __syncthreads();

        uint32_t a = s_state[0], b = s_state[1], c = s_state[2], d = s_state[3];
        uint32_t e = s_state[4], f = s_state[5], g = s_state[6], h = s_state[7];

        for (int t = 0; t < 64; ++t) {
            uint32_t S1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t temp1 = h + S1 + ch + d_sha256_k[t] + W[t];
            uint32_t S0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;
            h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
        }
        if (threadIdx.x == 0) {
            s_state[0] += a; s_state[1] += b; s_state[2] += c; s_state[3] += d;
            s_state[4] += e; s_state[5] += f; s_state[6] += g; s_state[7] += h;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        uint8_t* out = dev_out_hashes + seg_idx * out_stride;
        for (int i = 0; i < 8; ++i) {
            uint32_t v = s_state[i];
            out[i*4 + 0] = (v >> 24) & 0xFF;
            out[i*4 + 1] = (v >> 16) & 0xFF;
            out[i*4 + 2] = (v >> 8) & 0xFF;
            out[i*4 + 3] = v & 0xFF;
        }
    }
}

// Pairwise reduction kernel: hash pairs of 32-byte digests -> 32-byte digest
extern "C" __global__
void sha256_pair_reduce_kernel(const uint8_t* __restrict__ in_hashes, size_t in_count,
                               uint8_t* __restrict__ out_hashes) {
    size_t pair_idx = blockIdx.x;
    size_t i0 = pair_idx * 2;
    if (i0 >= in_count) return;

    // concatenate 32 + 32 bytes into 64-byte buffer
    __shared__ uint8_t buf[64];
    for (int i = threadIdx.x; i < 32; i += blockDim.x) {
        buf[i] = in_hashes[i0 * 32 + i];
    }
    if (i0 + 1 < in_count) {
        for (int i = threadIdx.x; i < 32; i += blockDim.x) {
            buf[32 + i] = in_hashes[(i0+1) * 32 + i];
        }
    } else {
        for (int i = threadIdx.x; i < 32; i += blockDim.x) {
            buf[32 + i] = 0;
        }
    }
    __syncthreads();

    // single-block SHA-256 on 64-byte buffer
    __shared__ uint32_t W[64];
    __shared__ uint32_t s_state[8];
    if (threadIdx.x == 0) {
        s_state[0] = 0x6a09e667u; s_state[1] = 0xbb67ae85u; s_state[2] = 0x3c6ef372u; s_state[3] = 0xa54ff53au;
        s_state[4] = 0x510e527fu; s_state[5] = 0x9b05688cu; s_state[6] = 0x1f83d9abu; s_state[7] = 0x5be0cd19u;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 16; i += blockDim.x) {
        W[i] = (uint32_t)buf[i*4 + 0] << 24 | (uint32_t)buf[i*4 + 1] << 16 | (uint32_t)buf[i*4 + 2] << 8 | (uint32_t)buf[i*4 + 3];
    }
    __syncthreads();

    // set length bits (512 bits)
    if (threadIdx.x == 0) {
        W[14] = 0;
        W[15] = 512;
    }
    __syncthreads();

    for (int t = 16 + threadIdx.x; t < 64; t += blockDim.x) {
        uint32_t s0 = rotr32(W[t-15],7) ^ rotr32(W[t-15],18) ^ (W[t-15] >> 3);
        uint32_t s1 = rotr32(W[t-2],17) ^ rotr32(W[t-2],19) ^ (W[t-2] >> 10);
        W[t] = W[t-16] + s0 + W[t-7] + s1;
    }
    __syncthreads();

    uint32_t a = s_state[0], b = s_state[1], c = s_state[2], d = s_state[3];
    uint32_t e = s_state[4], f = s_state[5], g = s_state[6], h = s_state[7];

    for (int t = 0; t < 64; ++t) {
        uint32_t S1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + d_sha256_k[t] + W[t];
        uint32_t S0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
    }
    if (threadIdx.x == 0) {
        uint32_t H0 = s_state[0] + a; uint32_t H1 = s_state[1] + b; uint32_t H2 = s_state[2] + c; uint32_t H3 = s_state[3] + d;
        uint32_t H4 = s_state[4] + e; uint32_t H5 = s_state[5] + f; uint32_t H6 = s_state[6] + g; uint32_t H7 = s_state[7] + h;
        uint8_t* out = out_hashes + pair_idx * 32;
        out[0] = (H0 >> 24) & 0xFF; out[1] = (H0 >> 16) & 0xFF; out[2] = (H0 >> 8) & 0xFF; out[3] = H0 & 0xFF;
        out[4] = (H1 >> 24) & 0xFF; out[5] = (H1 >> 16) & 0xFF; out[6] = (H1 >> 8) & 0xFF; out[7] = H1 & 0xFF;
        out[8] = (H2 >> 24) & 0xFF; out[9] = (H2 >> 16) & 0xFF; out[10] = (H2 >> 8) & 0xFF; out[11] = H2 & 0xFF;
        out[12] = (H3 >> 24) & 0xFF; out[13] = (H3 >> 16) & 0xFF; out[14] = (H3 >> 8) & 0xFF; out[15] = H3 & 0xFF;
        out[16] = (H4 >> 24) & 0xFF; out[17] = (H4 >> 16) & 0xFF; out[18] = (H4 >> 8) & 0xFF; out[19] = H4 & 0xFF;
        out[20] = (H5 >> 24) & 0xFF; out[21] = (H5 >> 16) & 0xFF; out[22] = (H5 >> 8) & 0xFF; out[23] = H5 & 0xFF;
        out[24] = (H6 >> 24) & 0xFF; out[25] = (H6 >> 16) & 0xFF; out[26] = (H6 >> 8) & 0xFF; out[27] = H6 & 0xFF;
        out[28] = (H7 >> 24) & 0xFF; out[29] = (H7 >> 16) & 0xFF; out[30] = (H7 >> 8) & 0xFF; out[31] = H7 & 0xFF;
    }
}

// Host-side orchestrator: compute tree-hash root hex string
static std::string gpu_tree_sha256_hex(uint8_t* d_buf, size_t len, cudaStream_t stream, size_t segment_size = (1<<20)) {
    if (len == 0) {
        // SHA256(empty) canonical value:
        return "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    }
    // clamp segment size for memory
    if (segment_size < 64) segment_size = 64;
    size_t num_segments = (len + segment_size - 1) / segment_size;
    // allocate device array for segment digests
    uint8_t* d_hashes;
    cudaMallocAsync(reinterpret_cast<void**>(&d_hashes), num_segments * 32, stream);

    int threads_per_block = 256;
    size_t shared_bytes = (64 + 8) * sizeof(uint32_t); // W[64] + state[8]
    sha256_segment_kernel<<<(int)num_segments, threads_per_block, (int)shared_bytes, stream>>>(d_buf, len, segment_size, d_hashes, 32);

    // iterative pairwise reduction
    size_t level_count = num_segments;
    uint8_t* d_curr = d_hashes;
    std::vector<uint8_t*> to_free;
    while (level_count > 1) {
        size_t next_count = (level_count + 1) / 2;
        uint8_t* d_next = nullptr;
        cudaMallocAsync(reinterpret_cast<void**>(&d_next), next_count * 32, stream);
        int blocks = (int)next_count;
        int threads = 128;
        sha256_pair_reduce_kernel<<<blocks, threads, 0, stream>>>(d_curr, level_count, d_next);
        to_free.push_back(d_curr);
        d_curr = d_next;
        level_count = next_count;
    }
    // copy final 32 bytes
    uint8_t final_hash[32];
    cudaMemcpyAsync(final_hash, d_curr, 32, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for (auto p : to_free) cudaFreeAsync(p, stream);
    cudaFreeAsync(d_curr, stream);

    std::ostringstream ss; ss << std::hex << std::setfill('0');
    for (int i = 0; i < 32; ++i) ss << std::setw(2) << (int)final_hash[i];
    return ss.str();
}

// ---------------------------
// nvCOMP batched zstd wrapper (uses nvCOMP low-level batched API)
// - Based on nvCOMP docs and examples for batched compress flow.
// - Important: check your nvCOMP header/version if symbol names differ.
// Docs described these batched functions: nvcompBatchedZstdCompressGetTempSizeAsync,
// nvcompBatchedZstdCompressGetMaxOutputChunkSize, nvcompBatchedZstdCompressAsync. :contentReference[oaicite:2]{index=2}
// ---------------------------

struct NvcompBatchResult {
    std::vector<std::vector<uint8_t>> compressed_chunks; // host copies
    std::vector<size_t> compressed_sizes;
    std::vector<std::string> chunk_shas; // hex (we'll compute via GPU tree-sha or device kernel)
    bool ok = false;
    std::string err;
};

// Compress a batch of device buffers (device_ptrs[]), each with device_sizes[] bytes.
// Returns compressed chunks on host. Uses provided CUDA stream.
static NvcompBatchResult nvcomp_batched_zstd_compress_device(void** device_ptrs, const size_t* device_sizes,
                                                             size_t num_chunks, size_t max_uncompressed_chunk_bytes,
                                                             cudaStream_t stream, int zstd_level = 3) {
    NvcompBatchResult result;
    if (num_chunks == 0) { result.ok = true; return result; }

    // Prepare batched options
    nvcompBatchedZstdCompressOpts_t opts = nvcompBatchedZstdCompressOptsDefault();
    opts.compression_level = zstd_level;

    // Query required temp workspace (device)
    size_t temp_bytes = 0;
    nvcompStatus_t st = nvcompBatchedZstdCompressGetTempSizeAsync(
        num_chunks, max_uncompressed_chunk_bytes, opts, &temp_bytes, max_uncompressed_chunk_bytes * num_chunks);
    if (st != nvcompSuccess) {
        result.err = "nvcomp: GetTempSizeAsync failed";
        return result;
    }

    // Query maximum output chunk size
    size_t max_out_chunk = 0;
    st = nvcompBatchedZstdCompressGetMaxOutputChunkSize(num_chunks, max_uncompressed_chunk_bytes, opts, &max_out_chunk);
    if (st != nvcompSuccess) {
        result.err = "nvcomp: GetMaxOutputChunkSize failed";
        return result;
    }

    // Allocate device temp workspace
    void* d_temp = nullptr;
    if (temp_bytes > 0) {
        if (cudaMallocAsync(&d_temp, temp_bytes, stream) != cudaSuccess) {
            result.err = "cudaMallocAsync temp failed";
            return result;
        }
    }

    // Allocate device output buffers for each chunk
    std::vector<void*> d_out_ptrs(num_chunks, nullptr);
    for (size_t i = 0; i < num_chunks; ++i) {
        if (cudaMallocAsync(&d_out_ptrs[i], max_out_chunk, stream) != cudaSuccess) {
            result.err = "cudaMallocAsync out buffer failed";
            for (size_t j = 0; j < i; ++j) cudaFreeAsync(d_out_ptrs[j], stream);
            if (d_temp) cudaFreeAsync(d_temp, stream);
            return result;
        }
    }

    // Allocate device arrays for output sizes and status
    size_t* d_out_sizes = nullptr;
    nvcompStatus_t* d_statuses = nullptr;
    cudaMallocAsync(reinterpret_cast<void**>(&d_out_sizes), sizeof(size_t) * num_chunks, stream);
    cudaMallocAsync(reinterpret_cast<void**>(&d_statuses), sizeof(nvcompStatus_t) * num_chunks, stream);

    // Launch batched compress async
    st = nvcompBatchedZstdCompressAsync(
        device_ptrs,
        device_sizes,
        max_uncompressed_chunk_bytes,
        num_chunks,
        d_temp,
        temp_bytes,
        d_out_ptrs.data(),
        d_out_sizes,
        opts,
        d_statuses,
        stream
    );

    if (st != nvcompSuccess) {
        result.err = "nvcomp: CompressAsync launch failed";
        for (auto p : d_out_ptrs) if (p) cudaFreeAsync(p, stream);
        if (d_temp) cudaFreeAsync(d_temp, stream);
        cudaFreeAsync(d_out_sizes, stream);
        cudaFreeAsync(d_statuses, stream);
        return result;
    }

    // Wait for completion
    cudaStreamSynchronize(stream);

    // copy compressed sizes & statuses back to host
    std::vector<size_t> out_sizes(num_chunks);
    std::vector<nvcompStatus_t> statuses(num_chunks);
    cudaMemcpy(out_sizes.data(), d_out_sizes, sizeof(size_t) * num_chunks, cudaMemcpyDeviceToHost);
    cudaMemcpy(statuses.data(), d_statuses, sizeof(nvcompStatus_t) * num_chunks, cudaMemcpyDeviceToHost);

    // For each chunk, copy compressed data to host and compute gpu tree-sha on device (we already have device compressed buffers)
    result.compressed_chunks.resize(num_chunks);
    result.compressed_sizes.resize(num_chunks);
    result.chunk_shas.resize(num_chunks);

    for (size_t i = 0; i < num_chunks; ++i) {
        if (statuses[i] != nvcompSuccess) {
            result.err = "nvcomp reported failure for chunk " + std::to_string(i);
            break;
        }
        size_t csize = out_sizes[i];
        result.compressed_sizes[i] = csize;
        // compute SHA on device buffer d_out_ptrs[i] using gpu tree-sha
        std::string hexsha = gpu_tree_sha256_hex(reinterpret_cast<uint8_t*>(d_out_ptrs[i]), csize, stream, 1<<20);
        result.chunk_shas[i] = hexsha;

        // copy compressed data to host
        std::vector<uint8_t> host_comp(csize);
        cudaMemcpyAsync(host_comp.data(), d_out_ptrs[i], csize, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        result.compressed_chunks[i] = std::move(host_comp);
    }

    // cleanup device allocs
    for (auto p : d_out_ptrs) if (p) cudaFreeAsync(p, stream);
    if (d_temp) cudaFreeAsync(d_temp, stream);
    if (d_out_sizes) cudaFreeAsync(d_out_sizes, stream);
    if (d_statuses) cudaFreeAsync(d_statuses, stream);

    if (!result.err.empty()) {
        result.ok = false;
    } else {
        result.ok = true;
    }
    return result;
}

// ---------------------------
// Simple thread-safe checkpoint manager
// ---------------------------
struct Checkpoint {
    std::string path;
    json j;
    std::mutex m;
    Checkpoint(const std::string& p): path(p) {
        if (std::filesystem::exists(p)) {
            std::ifstream in(p, std::ios::binary);
            if (in) in >> j;
        } else {
            j = json::object();
            j["done"] = json::object();
        }
    }
    void mark_done(const std::string& tensor_name) {
        std::lock_guard<std::mutex> lk(m);
        j["done"][tensor_name] = true;
        save();
    }
    bool is_done(const std::string& tensor_name) {
        std::lock_guard<std::mutex> lk(m);
        return j.contains("done") && j["done"].contains(tensor_name) && j["done"][tensor_name].get<bool>();
    }
    void save() {
        std::lock_guard<std::mutex> lk(m);
        std::ofstream o(path + ".tmp", std::ios::binary);
        o << j.dump(2);
        o.close();
        std::filesystem::rename(path + ".tmp", path);
    }
};

// ---------------------------
// SafeTensors streaming reader (robust)
// ---------------------------
class SafeTensorsStreamReader {
public:
    SafeTensorsStreamReader(const std::string& path) : path_(path) {
        file_.open(path, std::ios::binary);
        if (!file_) throw std::runtime_error("Cannot open safetensors: " + path);
        uint64_t header_len = 0;
        file_.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
        if (!file_) throw std::runtime_error("Failed to read header length");
        if (header_len == 0 || header_len > 200ULL * 1024 * 1024) throw std::runtime_error("Suspicious header length");
        header_len_ = header_len;
        std::string header(header_len_, '\0');
        file_.read(header.data(), header_len_);
        header_json_ = json::parse(header);
        data_offset_ = sizeof(header_len) + header_len_;
        for (auto& it : header_json_.items()) {
            if (it.key() == "__metadata__") continue;
            auto meta = it.value();
            uint64_t start = meta["data_offsets"][0].get<uint64_t>();
            uint64_t end = meta["data_offsets"][1].get<uint64_t>();
            offsets_[it.key()] = {start, end};
            dtypes_[it.key()] = meta.value("dtype", std::string("F32"));
            shapes_[it.key()] = meta["shape"].get<std::vector<int64_t>>();
        }
    }

    std::vector<std::tuple<std::string, std::vector<int64_t>, std::string, uint64_t>> enumerate_tensors_raw() {
        std::vector<std::tuple<std::string, std::vector<int64_t>, std::string, uint64_t>> out;
        for (auto& it : header_json_.items()) {
            if (it.key() == "__metadata__") continue;
            std::vector<int64_t> shape = shapes_[it.key()];
            std::string dtype = dtypes_[it.key()];
            uint64_t start = it.value()["data_offsets"][0].get<uint64_t>();
            uint64_t end   = it.value()["data_offsets"][1].get<uint64_t>();
            out.emplace_back(it.key(), shape, dtype, end - start);
        }
        return out;
    }

    size_t read_tensor_chunk(const std::string& tensor_name, uint64_t offset_in_tensor, uint8_t* buf, size_t bufsize) {
        auto it = offsets_.find(tensor_name);
        if (it == offsets_.end()) throw std::runtime_error("Unknown tensor: " + tensor_name);
        uint64_t start_offset = data_offset_ + it->second.first + offset_in_tensor;
        file_.seekg(start_offset, std::ios::beg);
        file_.read(reinterpret_cast<char*>(buf), bufsize);
        return file_.gcount();
    }

private:
    std::ifstream file_;
    std::string path_;
    uint64_t header_len_;
    uint64_t data_offset_;
    json header_json_;
    std::unordered_map<std::string, std::pair<uint64_t,uint64_t>> offsets_;
    std::unordered_map<std::string, std::string> dtypes_;
    std::unordered_map<std::string, std::vector<int64_t>> shapes_;
};

// ---------------------------
// GGUF writer (official API usage + streaming metadata)
// ---------------------------
class GGUFWriter {
public:
    GGUFWriter(const std::string& out_path) : out_path_(out_path) {
        ctx_ = gguf_init_empty();
        if (!ctx_) throw std::runtime_error("gguf_init_empty failed");
        gguf_set_val_str(ctx_, "general.name", "AdvancedGGUF_Converter_GPU_nvCOMP");
        gguf_set_val_str(ctx_, "general.architecture", "llama");
        gguf_set_val_str(ctx_, "general.version", "6.0");
    }
    ~GGUFWriter() { if (ctx_) gguf_free(ctx_); }

    void add_tensor_header(const std::string& name, const std::vector<int64_t>& shape, ggml_type ttype) {
        gguf_add_tensor(ctx_, name.c_str(), shape.data(), shape.size(), ttype, nullptr);
    }

    void write_header_initial() {
        int rc = gguf_write_to_file(ctx_, out_path_.c_str());
        if (rc != 0) throw std::runtime_error("gguf_write_to_file failed");
        append_file_ = std::fopen(out_path_.c_str(), "ab");
        if (!append_file_) throw std::runtime_error("Cannot open output for append");
    }

    uint64_t append_data(const uint8_t* data, size_t size) {
        std::lock_guard<std::mutex> lk(m_);
        uint64_t off = std::ftell(append_file_);
        size_t w = fwrite(data, 1, size, append_file_);
        if (w != size) throw std::runtime_error("Failed to append data");
        fflush(append_file_);
        return off;
    }

    void finalize_with_metadata(const json& meta) {
        gguf_set_val_str(ctx_, "gpu_streaming.metadata", meta.dump().c_str());
        if (append_file_) { fclose(append_file_); append_file_ = nullptr; }
        std::string tmp = out_path_ + ".tmp";
        int rc = gguf_write_to_file(ctx_, tmp.c_str());
        if (rc != 0) throw std::runtime_error("gguf_write_to_file(final) failed");
        std::filesystem::rename(tmp, out_path_);
    }

private:
    std::string out_path_;
    struct gguf_context* ctx_ = nullptr;
    FILE* append_file_ = nullptr;
    std::mutex m_;
};

// ---------------------------
// Converter core: GPU-first pipeline
// ---------------------------
class Converter {
public:
    Converter(const std::string& in_path, const std::string& out_path,
              size_t chunk_size = CHUNK_DEFAULT, unsigned int cuda_streams = 4)
    : reader_(in_path), writer_(out_path), chunk_size_(chunk_size), checkpoint_(CHECKPOINT_FILENAME) {
        cudaSetDevice(0);
        cuda_streams_ = std::max<unsigned int>(1, cuda_streams);
        for (unsigned int i = 0; i < cuda_streams_; ++i) {
            cudaStream_t s;
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
            streams_.push_back(s);
        }
    }
    ~Converter() {
        for (auto s : streams_) cudaStreamDestroy(s);
    }

    int run(unsigned int threads = 0) {
        try {
            auto tensors_raw = reader_.enumerate_tensors_raw();
            size_t n = tensors_raw.size();
            LOGI("Found %zu tensors", n);

            // Add gguf tensor headers
            for (auto &tup : tensors_raw) {
                const std::string &name = std::get<0>(tup);
                auto shape = std::get<1>(tup);
                std::string dtype = std::get<2>(tup);
                ggml_type ttype = parse_dtype(dtype);
                writer_.add_tensor_header(name, shape, ttype);
            }
            writer_.write_header_initial();

            // telemetry CSV header
            std::ofstream csv(TELEMETRY_CSV);
            csv << "tensor,orig_bytes,comp_bytes,ratio,ms,sha256\n";
            csv.close();
            json telemetry = json::array();

            std::atomic<size_t> index{0};
            unsigned int thread_count = threads == 0 ? std::max(1u, std::thread::hardware_concurrency()) : threads;
            std::vector<std::thread> workers;
            for (unsigned int t = 0; t < thread_count; ++t) {
                workers.emplace_back([&, t]() {
                    size_t my;
                    while ((my = index.fetch_add(1)) < n && !g_cancelled) {
                        process_one_tensor(tensors_raw[my], telemetry, t % cuda_streams_);
                    }
                });
            }
            for (auto &w : workers) w.join();

            if (g_cancelled) { LOGW("Cancelled; checkpoint saved"); return 1; }

            // finalize GGUF metadata
            json meta; meta["tensors"] = json::object();
            {
                std::lock_guard<std::mutex> lk(completed_mutex_);
                for (auto &entry : completed_tensors_) {
                    json tj;
                    tj["orig_bytes"] = entry.second.original_bytes;
                    tj["comp_bytes"] = entry.second.compressed_total;
                    tj["chunks"] = json::array();
                    for (auto &c : entry.second.chunks) {
                        json cj;
                        cj["orig_offset"] = c.orig_offset;
                        cj["orig_size"] = c.orig_size;
                        cj["comp_size"] = c.comp_size;
                        cj["comp_sha256"] = c.comp_sha256;
                        cj["out_file_offset"] = c.out_file_offset;
                        tj["chunks"].push_back(cj);
                    }
                    meta["tensors"][entry.first] = tj;
                }
            }

            writer_.finalize_with_metadata(meta);

            // telemetry file
            std::ofstream tj(TELEMETRY_JSON);
            tj << telemetry.dump(2);
            tj.close();

            LOGI("Conversion completed successfully");
            return 0;
        } catch (const std::exception &e) {
            LOGE("FATAL: %s", e.what());
            return 1;
        }
    }

private:
    struct ChunkInfo { uint64_t orig_offset; size_t orig_size; size_t comp_size; std::string comp_sha256; uint64_t out_file_offset; };
    struct LocalTensor { std::vector<ChunkInfo> chunks; uint64_t original_bytes = 0; size_t compressed_total = 0; double compress_time_ms = 0.0; };

    void process_one_tensor(const std::tuple<std::string, std::vector<int64_t>, std::string, uint64_t>& tup,
                            json& telemetry_out, unsigned int stream_id) {
        const std::string &name = std::get<0>(tup);
        uint64_t original_bytes = std::get<3>(tup);
        LOGI("Processing tensor %s (%.2f MB)", name.c_str(), original_bytes / 1e6);
        if (checkpoint_.is_done(name)) { LOGI("Skipping %s (checkpoint)", name.c_str()); return; }

        LocalTensor lt; lt.original_bytes = original_bytes;
        uint64_t remaining = original_bytes;
        uint64_t offset = 0;
        auto tstart = std::chrono::high_resolution_clock::now();

        // Process in chunk_size_ slices
        while (remaining > 0 && !g_cancelled) {
            size_t use = (size_t)std::min<uint64_t>(chunk_size_, remaining);

            // pinned host read
            uint8_t* h_buf = nullptr;
            if (cudaMallocHost(reinterpret_cast<void**>(&h_buf), use) != cudaSuccess) throw std::runtime_error("cudaMallocHost failed");
            size_t got = reader_.read_tensor_chunk(name, offset, h_buf, use);
            if (got == 0) { cudaFreeHost(h_buf); throw std::runtime_error("Unexpected EOF reading tensor chunk"); }

            // device staging
            uint8_t* d_buf = nullptr;
            if (cudaMallocAsync(reinterpret_cast<void**>(&d_buf), got, streams_[stream_id]) != cudaSuccess) {
                cudaFreeHost(h_buf); throw std::runtime_error("cudaMallocAsync device input failed");
            }
            cudaMemcpyAsync(d_buf, h_buf, got, cudaMemcpyHostToDevice, streams_[stream_id]);

            // call nvCOMP batched compress for single-chunk (we can batch multiple chunks by collecting inputs; here we use single)
            void* dev_ptrs[1] = { d_buf };
            size_t dev_sizes[1] = { got };
            NvcompBatchResult br = nvcomp_batched_zstd_compress_device(dev_ptrs, dev_sizes, 1, got, streams_[stream_id], /*zstd_level*/ 3);

            // free pinned input and device input
            cudaFreeHost(h_buf);
            cudaFreeAsync(d_buf, streams_[stream_id]);

            if (!br.ok) {
                LOGW("nvCOMP failed for %s chunk -> fallback to CPU ZSTD", name.c_str());
                // CPU fallback: re-read into host and compress
                std::vector<uint8_t> raw(use);
                size_t got2 = reader_.read_tensor_chunk(name, offset, raw.data(), use);
                if (got2 == 0) throw std::runtime_error("Fallback read failed");
                size_t bound = ZSTD_compressBound(got2);
                std::vector<uint8_t> out(bound);
                int level = std::getenv("COMPRESSION_LEVEL") ? std::atoi(std::getenv("COMPRESSION_LEVEL")) : 3;
                size_t csize = ZSTD_compress(out.data(), bound, raw.data(), got2, level);
                if (ZSTD_isError(csize)) throw std::runtime_error("ZSTD fallback failed");
                out.resize(csize);
                std::string sha = sha256_hex_cpu(out.data(), out.size());
                uint64_t file_off = writer_.append_data(out.data(), out.size());
                ChunkInfo ci{offset, (size_t)got2, csize, sha, file_off};
                lt.chunks.push_back(ci);
                lt.compressed_total += csize;
            } else {
                // use first compressed chunk from nvCOMP result
                auto &comp = br.compressed_chunks[0];
                size_t csize = br.compressed_sizes[0];
                std::string sha = br.chunk_shas[0]; // computed on-device in nvCOMP wrapper
                uint64_t file_off = writer_.append_data(comp.data(), csize);
                ChunkInfo ci{offset, (size_t)got, csize, sha, file_off};
                lt.chunks.push_back(ci);
                lt.compressed_total += csize;
            }

            offset += use;
            remaining -= use;
        } // end while chunks

        auto tend = std::chrono::high_resolution_clock::now();
        lt.compress_time_ms = std::chrono::duration<double, std::milli>(tend - tstart).count();

        // record completed tensor
        {
            std::lock_guard<std::mutex> lk(completed_mutex_);
            completed_tensors_[name] = lt;
        }

        // telemetry write
        {
            std::string joinsha;
            for (auto &c : lt.chunks) joinsha += c.comp_sha256;
            std::string tensor_sha = sha256_hex_cpu((const uint8_t*)joinsha.data(), joinsha.size());
            std::ofstream csv(TELEMETRY_CSV, std::ios::app);
            csv << name << "," << lt.original_bytes << "," << lt.compressed_total << ","
                << (lt.original_bytes ? (double)lt.original_bytes / (double)lt.compressed_total : 1.0) << ","
                << lt.compress_time_ms << "," << tensor_sha << "\n";
            csv.close();
            json tj;
            tj["name"] = name; tj["orig_bytes"] = lt.original_bytes; tj["comp_bytes"] = lt.compressed_total;
            tj["ratio"] = lt.original_bytes ? (double)lt.original_bytes / (double)lt.compressed_total : 1.0;
            tj["time_ms"] = lt.compress_time_ms; tj["sha"] = tensor_sha;
            telemetry_out.push_back(tj);
        }

        checkpoint_.mark_done(name);
    }

    ggml_type parse_dtype(const std::string& s) {
        static std::unordered_map<std::string, ggml_type> m = {
            {"F32", GGML_TYPE_F32}, {"float32", GGML_TYPE_F32},
            {"F16", GGML_TYPE_F16}, {"float16", GGML_TYPE_F16},
            {"BF16", GGML_TYPE_BF16}, {"bfloat16", GGML_TYPE_BF16},
            {"I32", GGML_TYPE_I32}, {"int32", GGML_TYPE_I32},
            {"I16", GGML_TYPE_I16}, {"int16", GGML_TYPE_I16},
            {"I8", GGML_TYPE_I8}, {"int8", GGML_TYPE_I8},
            {"U8", GGML_TYPE_I8}, {"uint8", GGML_TYPE_I8}
        };
        auto it = m.find(s);
        if (it == m.end()) return GGML_TYPE_F32;
        return it->second;
    }

    SafeTensorsStreamReader reader_;
    GGUFWriter writer_;
    size_t chunk_size_;
    Checkpoint checkpoint_;
    std::vector<cudaStream_t> streams_;
    unsigned int cuda_streams_;
    std::mutex completed_mutex_;
    std::unordered_map<std::string, LocalTensor> completed_tensors_;
};

// ---------------------------
// main
// ---------------------------
int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.safetensors> <output.gguf> [chunk_bytes] [cuda_streams] [threads]\n", argv[0]);
        return 1;
    }
#ifdef _WIN32
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
#else
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
#endif

    std::string input = argv[1];
    std::string output = argv[2];
    size_t chunk = CHUNK_DEFAULT;
    unsigned int streams = 4;
    unsigned int threads = 0;
    if (argc >= 4) chunk = std::min<uint64_t>(CHUNK_MAX, std::max<uint64_t>(CHUNK_MIN, std::stoull(argv[3])));
    if (argc >= 5) streams = std::max<unsigned int>(1, std::stoul(argv[4]));
    if (argc >= 6) threads = std::stoul(argv[5]);

    try {
        LOGI("AdvancedGGUF_Converter v6.0 starting (nvCOMP GPU + GPU tree-SHA256)");
        LOGI("Input: %s, Output: %s, Chunk: %zu, CUDA streams: %u, Threads: %u", input.c_str(), output.c_str(), chunk, streams, threads);
        Converter conv(input, output, chunk, streams);
        int rc = conv.run(threads);
        if (g_cancelled) { LOGW("Cancelled."); return 1; }
        return rc;
    } catch (const std::exception &e) {
        LOGE("FATAL: %s", e.what());
        return 1;
    }
}
'@
}
