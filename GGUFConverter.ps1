function Get-EnhancedConverterSource {
    return @'
/*
  AdvancedGGUF_Converter v7.0 — High-Performance GPU-First Conversion
  - True batched nvCOMP ZSTD for maximum GPU throughput.
  - Multi-stream GPU tree-based SHA256 for integrity checks.
  - Modern C++ RAII for robust, leak-free memory management.
  - Drop-in single-file C++ source. Compile with nvcc or a host compiler linked against CUDA/nvCOMP.

  Key Improvements (v7.0):
  1. Performance: Implemented true batching for nvCOMP to process multiple data chunks in a single GPU launch, maximizing hardware utilization.
  2. Memory Safety: Replaced raw pointers and manual cudaMalloc/Free with std::unique_ptr and custom deleters (RAII), preventing memory leaks.
  3. Code Quality: Refactored into namespaces for better organization and readability. Enhanced error handling with a custom exception class.
  4. Clarity: Added extensive comments explaining the batching pipeline, GPU algorithms, and overall architecture.

  Requires:
  - CUDA Toolkit (Runtime & Driver)
  - nvCOMP Library (Batched API)
  - GGUF/GGML headers & library
  - Zstandard library (for CPU fallback)
  - nlohmann::json (for metadata)
  - OpenSSL (for CPU SHA256 fallback)

  Example Build (Linux):
      nvcc -O3 -std=c++17 -Xcompiler -fPIC converter.cpp -o converter \
           -I/path/to/json/include -I/path/to/gguf/include -I/usr/local/cuda/include -I/path/to/nvcomp/include \
           -L/path/to/gguf/lib -L/usr/local/cuda/lib64 -L/path/to/nvcomp/lib \
           -lnvcomp -lcudart -lggml -lgguf -lzstd -lssl -lcrypto
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

// Required libraries
#include "json.hpp"
using json = nlohmann::json;

#define GGML_BUILD
#include "ggml.h"
#include "gguf.h"

#include <cuda_runtime.h>
#include <nvcomp.h>
#include <zstd.h>
#include <openssl/sha.h>

// Optional DFloat11 API for CPU fallback
extern "C" {
    size_t DFloat11_compress_bound(size_t size);
    int    DFloat11_compress(const uint8_t* src, size_t src_size, uint8_t* dst, size_t* dst_size);
    int    DFloat11_decompress(const uint8_t* src, size_t src_size, uint8_t* dst, size_t dst_size);
}

// ----------------------------------------------------------------------------
// Global Configuration
// ----------------------------------------------------------------------------
static constexpr size_t CHUNK_MIN_SIZE_BYTES    = 4ULL * 1024 * 1024;        // 4 MiB
static constexpr size_t CHUNK_DEFAULT_SIZE_BYTES= 16ULL * 1024 * 1024;     // 16 MiB
static constexpr size_t CHUNK_MAX_SIZE_BYTES    = 128ULL * 1024 * 1024;    // 128 MiB (adjust based on VRAM)

// Configuration for batching chunks before submitting to GPU
static constexpr size_t BATCH_MAX_CHUNKS          = 16;   // Max number of chunks in a single batch
static constexpr size_t BATCH_MAX_BYTES_UNCOMPRESSED = 256ULL * 1024 * 1024; // Max total size of a batch

static constexpr const char* CHECKPOINT_FILENAME = "convert_checkpoint.json";
static constexpr const char* TELEMETRY_CSV_FILENAME = "convert_telemetry.csv";
static constexpr const char* TELEMETRY_JSON_FILENAME = "convert_telemetry.json";

// ----------------------------------------------------------------------------
// Utilities: Logging, Exceptions, RAII, Signals
// ----------------------------------------------------------------------------

// Custom exception for detailed error reporting
class ConversionException : public std::runtime_error {
public:
    ConversionException(const std::string& message) : std::runtime_error(message) {}
};

// Thread-safe logger
enum class LogLevel { DEBUG, INFO, WARN, ERROR };
static std::mutex g_log_mutex;
static void log_message(LogLevel level, const char* fmt, ...) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    const char* names[] = {"DEBUG", "INFO", "WARN", "ERROR"};
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    fprintf(stderr, "[%s.%03lld] [%s] ", buf, (long long)ms.count(), names[(int)level]);
    va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fprintf(stderr, "\n"); fflush(stderr);
}
#define LOGD(...) log_message(LogLevel::DEBUG, __VA_ARGS__)
#define LOGI(...) log_message(LogLevel::INFO,  __VA_ARGS__)
#define LOGW(...) log_message(LogLevel::WARN,  __VA_ARGS__)
#define LOGE(...) log_message(LogLevel::ERROR, __VA_ARGS__)

// Signal handling for graceful cancellation
static std::atomic<bool> g_cancelled{false};
#ifdef _WIN32
#include <windows.h>
BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT) {
        LOGI("Cancellation requested by user, will shut down gracefully...");
        g_cancelled = true;
        return TRUE;
    }
    return FALSE;
}
#else
#include <signal.h>
void signal_handler(int s) {
    LOGI("Signal %d received, cancellation requested...", s);
    g_cancelled = true;
}
#endif

// RAII wrappers for CUDA memory management
namespace CudaUtils {
    struct DeviceDeleter { void operator()(void* p) { if (p) cudaFree(p); } };
    struct HostDeleter { void operator()(void* p) { if (p) cudaFreeHost(p); } };

    template<typename T> using unique_device_ptr = std::unique_ptr<T, DeviceDeleter>;
    template<typename T> using unique_host_ptr = std::unique_ptr<T, HostDeleter>;

    template<typename T>
    unique_device_ptr<T> make_unique_device(size_t size, cudaStream_t stream) {
        void* ptr = nullptr;
        if (cudaMallocAsync(&ptr, size, stream) != cudaSuccess) {
            throw ConversionException("cudaMallocAsync for device memory failed");
        }
        return unique_device_ptr<T>(static_cast<T*>(ptr));
    }

    template<typename T>
    unique_host_ptr<T> make_unique_host(size_t size) {
        void* ptr = nullptr;
        if (cudaMallocHost(&ptr, size) != cudaSuccess) {
            throw ConversionException("cudaMallocHost for pinned memory failed");
        }
        return unique_host_ptr<T>(static_cast<T*>(ptr));
    }
} // namespace CudaUtils

// ----------------------------------------------------------------------------
// CPU Fallback Utilities
// ----------------------------------------------------------------------------
namespace CpuUtils {
    // Fallback SHA256 calculation using OpenSSL
    std::string sha256_hex(const uint8_t* data, size_t len) {
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256(data, len, hash);
        std::ostringstream ss;
        ss << std::hex << std::setfill('0');
        for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
            ss << std::setw(2) << static_cast<int>(hash[i]);
        }
        return ss.str();
    }
} // namespace CpuUtils

// ----------------------------------------------------------------------------
// GPU SHA256 Implementation (Tree-Hash)
// ----------------------------------------------------------------------------
namespace GpuSha256 {
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
            s_state[0] = 0x6a09e667u; s_state[1] = 0xbb67ae85u; s_state[2] = 0x3c6ef372u; s_state[3] = 0xa54ff53au;
            s_state[4] = 0x510e527fu; s_state[5] = 0x9b05688cu; s_state[6] = 0x1f83d9abu; s_state[7] = 0x5be0cd19u;
        }
        __syncthreads();

        size_t num_blocks = (this_len + 8 + 63) / 64;
        for (size_t blk = 0; blk < num_blocks; ++blk) {
            for (int i = threadIdx.x; i < 16; i += blockDim.x) {
                size_t byte_index = blk * 64 + i * 4;
                uint32_t val = 0;
                for (int b = 0; b < 4; ++b) {
                    size_t idx = byte_index + b;
                    uint8_t byte = (idx < this_len) ? seg_ptr[idx] : ((idx == this_len) ? 0x80 : 0);
                    val = (val << 8) | byte;
                }
                W[i] = val;
            }
            __syncthreads();

            if (blk == num_blocks - 1) {
                if (threadIdx.x == 0) {
                    uint64_t bits_len = (uint64_t)this_len * 8;
                    W[14] = (uint32_t)(bits_len >> 32);
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
                out[i*4 + 0] = (v >> 24) & 0xFF; out[i*4 + 1] = (v >> 16) & 0xFF;
                out[i*4 + 2] = (v >> 8)  & 0xFF; out[i*4 + 3] = v & 0xFF;
            }
        }
    }

    // Pairwise reduction kernel: hash pairs of 32-byte digests -> 32-byte digest
    extern "C" __global__
    void sha256_pair_reduce_kernel(const uint8_t* __restrict__ in_hashes, size_t in_count,
                                   uint8_t* __restrict__ out_hashes) {
        size_t pair_idx = blockIdx.x;
        if (pair_idx * 2 >= in_count) return;

        __shared__ uint8_t buf[64];
        for (int i = threadIdx.x; i < 64; i += blockDim.x) {
            size_t src_idx = (pair_idx * 2 * 32) + i;
            if (src_idx < in_count * 32) {
                buf[i] = in_hashes[src_idx];
            } else {
                buf[i] = (i == 32) ? 0x80 : 0; // Padding for the second hash if it's an odd one out
            }
        }
        __syncthreads();

        __shared__ uint32_t W[16];
        __shared__ uint32_t s_state[8];
        if (threadIdx.x == 0) {
            s_state[0] = 0x6a09e667u; s_state[1] = 0xbb67ae85u; s_state[2] = 0x3c6ef372u; s_state[3] = 0xa54ff53au;
            s_state[4] = 0x510e527fu; s_state[5] = 0x9b05688cu; s_state[6] = 0x1f83d9abu; s_state[7] = 0x5be0cd19u;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < 16; i += blockDim.x) {
            W[i] = (uint32_t)buf[i*4+0]<<24 | (uint32_t)buf[i*4+1]<<16 | (uint32_t)buf[i*4+2]<<8 | (uint32_t)buf[i*4+3];
        }
        __syncthreads();

        uint32_t a = s_state[0], b = s_state[1], c = s_state[2], d = s_state[3];
        uint32_t e = s_state[4], f = s_state[5], g = s_state[6], h = s_state[7];

        for (int t = 0; t < 64; ++t) {
            uint32_t temp_w;
            if (t < 16) {
                temp_w = W[t];
            } else {
                uint32_t s0 = rotr32(W[(t-15)&15],7) ^ rotr32(W[(t-15)&15],18) ^ (W[(t-15)&15] >> 3);
                uint32_t s1 = rotr32(W[(t-2)&15],17) ^ rotr32(W[(t-2)&15],19) ^ (W[(t-2)&15] >> 10);
                temp_w = W[(t-16)&15] + s0 + W[(t-7)&15] + s1;
                if (threadIdx.x == (t % blockDim.x)) W[t&15] = temp_w;
            }
            __syncthreads();

            uint32_t S1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t temp1 = h + S1 + ch + d_sha256_k[t] + temp_w;
            uint32_t S0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;
            h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
        }

        if (threadIdx.x == 0) {
            uint32_t H[] = {s_state[0]+a, s_state[1]+b, s_state[2]+c, s_state[3]+d, s_state[4]+e, s_state[5]+f, s_state[6]+g, s_state[7]+h};
            uint8_t* out = out_hashes + pair_idx * 32;
            for(int i=0; i<8; ++i) {
                out[i*4+0] = (H[i]>>24)&0xFF; out[i*4+1] = (H[i]>>16)&0xFF; out[i*4+2] = (H[i]>>8)&0xFF; out[i*4+3] = H[i]&0xFF;
            }
        }
    }

    // Host function to orchestrate the tree-hash computation on the GPU
    std::string compute_tree_hash_hex(uint8_t* d_buf, size_t len, cudaStream_t stream, size_t segment_size = (1 << 20)) {
        if (len == 0) return "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

        segment_size = std::max<size_t>(64, segment_size);
        size_t num_segments = (len + segment_size - 1) / segment_size;

        auto d_hashes = CudaUtils::make_unique_device<uint8_t>(num_segments * 32, stream);

        int threads_per_block = 256;
        size_t shared_bytes = (64 + 8) * sizeof(uint32_t);
        sha256_segment_kernel<<<(int)num_segments, threads_per_block, (int)shared_bytes, stream>>>(d_buf, len, segment_size, d_hashes.get(), 32);

        size_t level_count = num_segments;
        CudaUtils::unique_device_ptr<uint8_t> d_curr = std::move(d_hashes);

        while (level_count > 1) {
            size_t next_count = (level_count + 1) / 2;
            auto d_next = CudaUtils::make_unique_device<uint8_t>(next_count * 32, stream);
            sha256_pair_reduce_kernel<<<(int)next_count, 128, 0, stream>>>(d_curr.get(), level_count, d_next.get());
            d_curr = std::move(d_next);
            level_count = next_count;
        }

        uint8_t final_hash[32];
        cudaMemcpyAsync(final_hash, d_curr.get(), 32, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        std::ostringstream ss;
        ss << std::hex << std::setfill('0');
        for (int i = 0; i < 32; ++i) ss << std::setw(2) << static_cast<int>(final_hash[i]);
        return ss.str();
    }
} // namespace GpuSha256

// ----------------------------------------------------------------------------
// nvCOMP GPU ZSTD Compression Wrapper
// ----------------------------------------------------------------------------
namespace NvcompGpuZstd {
    struct BatchResult {
        std::vector<std::vector<uint8_t>> compressed_chunks;
        std::vector<std::string> chunk_shas_hex;
    };

    // Compresses a batch of device buffers using the nvCOMP low-level batched API.
    BatchResult compress_batch(
        const std::vector<void*>& d_in_ptrs,
        const std::vector<size_t>& in_sizes,
        size_t max_uncompressed_chunk_bytes,
        cudaStream_t stream,
        int zstd_level = 3
    ) {
        size_t num_chunks = d_in_ptrs.size();
        if (num_chunks == 0) return {};

        nvcompBatchedZstdCompressOpts_t opts = nvcompBatchedZstdCompressOptsDefault();
        opts.compression_level = zstd_level;

        size_t temp_bytes = 0;
        nvcompStatus_t st = nvcompBatchedZstdCompressGetTempSize(num_chunks, max_uncompressed_chunk_bytes, opts, &temp_bytes);
        if (st != nvcompSuccess) throw ConversionException("nvcompBatchedZstdCompressGetTempSize failed");

        size_t max_out_chunk_bytes = 0;
        st = nvcompBatchedZstdCompressGetMaxOutputChunkSize(max_uncompressed_chunk_bytes, opts, &max_out_chunk_bytes);
        if (st != nvcompSuccess) throw ConversionException("nvcompBatchedZstdCompressGetMaxOutputChunkSize failed");

        auto d_temp = CudaUtils::make_unique_device<uint8_t>(temp_bytes, stream);
        auto d_out_sizes = CudaUtils::make_unique_device<size_t>(num_chunks * sizeof(size_t), stream);
        auto d_statuses = CudaUtils::make_unique_device<nvcompStatus_t>(num_chunks * sizeof(nvcompStatus_t), stream);
        
        std::vector<CudaUtils::unique_device_ptr<uint8_t>> d_out_buffers;
        std::vector<void*> d_out_ptrs;
        d_out_buffers.reserve(num_chunks);
        d_out_ptrs.reserve(num_chunks);

        for (size_t i = 0; i < num_chunks; ++i) {
            d_out_buffers.push_back(CudaUtils::make_unique_device<uint8_t>(max_out_chunk_bytes, stream));
            d_out_ptrs.push_back(d_out_buffers.back().get());
        }

        st = nvcompBatchedZstdCompressAsync(
            d_in_ptrs.data(), in_sizes.data(), max_uncompressed_chunk_bytes, num_chunks,
            d_temp.get(), temp_bytes, d_out_ptrs.data(), d_out_sizes.get(), opts, d_statuses.get(), stream);
        if (st != nvcompSuccess) throw ConversionException("nvcompBatchedZstdCompressAsync launch failed");

        std::vector<size_t> out_sizes(num_chunks);
        std::vector<nvcompStatus_t> statuses(num_chunks);
        cudaMemcpyAsync(out_sizes.data(), d_out_sizes.get(), sizeof(size_t) * num_chunks, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(statuses.data(), d_statuses.get(), sizeof(nvcompStatus_t) * num_chunks, cudaMemcpyDeviceToHost, stream);

        BatchResult result;
        result.compressed_chunks.resize(num_chunks);
        result.chunk_shas_hex.resize(num_chunks);

        for (size_t i = 0; i < num_chunks; ++i) {
            result.chunk_shas_hex[i] = GpuSha256::compute_tree_hash_hex(
                reinterpret_cast<uint8_t*>(d_out_ptrs[i]), out_sizes[i], stream
            );
        }

        cudaStreamSynchronize(stream);

        for (size_t i = 0; i < num_chunks; ++i) {
            if (statuses[i] != nvcompSuccess) {
                throw ConversionException("nvCOMP compression failed for chunk " + std::to_string(i));
            }
            result.compressed_chunks[i].resize(out_sizes[i]);
            cudaMemcpy(result.compressed_chunks[i].data(), d_out_ptrs[i], out_sizes[i], cudaMemcpyDeviceToHost);
        }

        return result;
    }
} // namespace NvcompGpuZstd

// ----------------------------------------------------------------------------
// I/O and Checkpointing Classes
// ----------------------------------------------------------------------------
class Checkpoint {
    std::string path_;
    json data_;
    std::mutex mutex_;
public:
    Checkpoint(const std::string& p) : path_(p) {
        if (std::filesystem::exists(p)) {
            std::ifstream in(p, std::ios::binary);
            if (in) try { in >> data_; } catch(...) { data_ = json::object(); }
        }
        if (!data_.is_object() || !data_.contains("completed_tensors")) {
            data_ = json::object();
            data_["completed_tensors"] = json::object();
        }
    }
    void mark_done(const std::string& tensor_name) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_["completed_tensors"][tensor_name] = true;
        save_internal();
    }
    bool is_done(const std::string& tensor_name) {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_["completed_tensors"].contains(tensor_name);
    }
private:
    void save_internal() {
        std::ofstream out(path_ + ".tmp", std::ios::binary);
        out << data_.dump(2);
        out.close();
        std::filesystem::rename(path_ + ".tmp", path_);
    }
};

class SafeTensorsStreamReader {
    std::ifstream file_;
    uint64_t data_offset_;
    json header_json_;
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> offsets_;
public:
    SafeTensorsStreamReader(const std::string& path) {
        file_.open(path, std::ios::binary);
        if (!file_) throw ConversionException("Cannot open safetensors file: " + path);
        uint64_t header_len = 0;
        file_.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
        if (!file_ || header_len == 0 || header_len > 200ULL * 1024 * 1024) {
            throw ConversionException("Invalid or corrupt safetensors header");
        }
        std::string header_str(header_len, '\0');
        file_.read(header_str.data(), header_len);
        header_json_ = json::parse(header_str);
        data_offset_ = sizeof(header_len) + header_len;
        for (auto& [key, value] : header_json_.items()) {
            if (key == "__metadata__") continue;
            offsets_[key] = {value["data_offsets"][0].get<uint64_t>(), value["data_offsets"][1].get<uint64_t>()};
        }
    }

    std::vector<std::tuple<std::string, std::vector<int64_t>, std::string, uint64_t>> enumerate_tensors() {
        std::vector<std::tuple<std::string, std::vector<int64_t>, std::string, uint64_t>> out;
        for (auto& [key, value] : header_json_.items()) {
            if (key == "__metadata__") continue;
            auto& offsets = offsets_.at(key);
            out.emplace_back(
                key,
                value["shape"].get<std::vector<int64_t>>(),
                value.value("dtype", "F32"),
                offsets.second - offsets.first
            );
        }
        return out;
    }

    size_t read_tensor_chunk(const std::string& name, uint64_t offset_in_tensor, uint8_t* buf, size_t size) {
        auto it = offsets_.find(name);
        if (it == offsets_.end()) throw ConversionException("Unknown tensor: " + name);
        uint64_t file_pos = data_offset_ + it->second.first + offset_in_tensor;
        file_.seekg(file_pos);
        file_.read(reinterpret_cast<char*>(buf), size);
        return file_.gcount();
    }
};

class GGUFWriter {
    std::string out_path_;
    struct gguf_context* ctx_ = nullptr;
    FILE* append_file_ = nullptr;
    std::mutex mutex_;
public:
    GGUFWriter(const std::string& path) : out_path_(path) {
        ctx_ = gguf_init_empty();
        if (!ctx_) throw ConversionException("gguf_init_empty failed");
        gguf_set_val_str(ctx_, "general.architecture", "llama");
    }
    ~GGUFWriter() {
        if (append_file_) fclose(append_file_);
        if (ctx_) gguf_free(ctx_);
    }

    void add_tensor_header(const std::string& name, const std::vector<int64_t>& shape, ggml_type ttype) {
        gguf_add_tensor(ctx_, name.c_str(), shape.data(), shape.size(), ttype, nullptr);
    }

    void write_initial_header() {
        gguf_write_to_file(ctx_, out_path_.c_str());
        append_file_ = std::fopen(out_path_.c_str(), "ab");
        if (!append_file_) throw ConversionException("Cannot open output file for appending");
    }

    uint64_t append_data(const uint8_t* data, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        uint64_t offset = std::ftell(append_file_);
        if (fwrite(data, 1, size, append_file_) != size) {
            throw ConversionException("Failed to write data to GGUF file");
        }
        fflush(append_file_);
        return offset;
    }

    void finalize(const json& metadata) {
        gguf_set_val_str(ctx_, "gpu_streaming.metadata", metadata.dump().c_str());
        if (append_file_) { fclose(append_file_); append_file_ = nullptr; }
        std::string tmp_path = out_path_ + ".tmp";
        gguf_write_to_file(ctx_, tmp_path.c_str());
        std::filesystem::rename(tmp_path, out_path_);
    }
};

// ----------------------------------------------------------------------------
// Converter Core Logic
// ----------------------------------------------------------------------------
class Converter {
public:
    Converter(const std::string& in_path, const std::string& out_path, size_t chunk_size, unsigned int num_streams)
    : reader_(in_path), writer_(out_path), chunk_size_(chunk_size), checkpoint_(CHECKPOINT_FILENAME) {
        cudaSetDevice(0);
        for (unsigned int i = 0; i < num_streams; ++i) {
            cudaStream_t s;
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
            streams_.push_back(s);
        }
    }
    ~Converter() {
        for (auto s : streams_) cudaStreamDestroy(s);
    }

    int run(unsigned int num_threads) {
        auto tensors = reader_.enumerate_tensors();
        LOGI("Found %zu tensors in the input file.", tensors.size());

        for (const auto& [name, shape, dtype, size] : tensors) {
            writer_.add_tensor_header(name, shape, parse_dtype(dtype));
        }
        writer_.write_initial_header();

        std::ofstream csv(TELEMETRY_CSV_FILENAME);
        csv << "tensor_name,original_bytes,compressed_bytes,ratio,time_ms,final_sha256\n";
        csv.close();
        json telemetry_json = json::array();

        std::atomic<size_t> tensor_idx{0};
        std::vector<std::thread> workers;
        for (unsigned int i = 0; i < num_threads; ++i) {
            workers.emplace_back([&, i]() {
                size_t idx;
                while (!g_cancelled && (idx = tensor_idx.fetch_add(1)) < tensors.size()) {
                    process_tensor(tensors[idx], telemetry_json, i % streams_.size());
                }
            });
        }
        for (auto& w : workers) w.join();

        if (g_cancelled) {
            LOGW("Conversion cancelled. Checkpoint has been saved.");
            return 1;
        }

        json final_meta;
        final_meta["tensors"] = json::object();
        {
            std::lock_guard<std::mutex> lock(completed_mutex_);
            for (auto& [name, data] : completed_tensors_) {
                json tj;
                tj["original_bytes"] = data.original_bytes;
                tj["compressed_bytes"] = data.compressed_total;
                tj["chunks"] = json::array();
                for (auto& c : data.chunks) {
                    tj["chunks"].push_back({
                        {"orig_offset", c.orig_offset}, {"orig_size", c.orig_size},
                        {"comp_size", c.comp_size}, {"comp_sha256", c.comp_sha256},
                        {"out_file_offset", c.out_file_offset}
                    });
                }
                final_meta["tensors"][name] = tj;
            }
        }
        writer_.finalize(final_meta);

        std::ofstream tjson(TELEMETRY_JSON_FILENAME);
        tjson << telemetry_json.dump(2);
        tjson.close();

        LOGI("Conversion completed successfully.");
        return 0;
    }

private:
    struct ChunkInfo { uint64_t orig_offset; size_t orig_size; size_t comp_size; std::string comp_sha256; uint64_t out_file_offset; };
    struct TensorResult { std::vector<ChunkInfo> chunks; uint64_t original_bytes = 0; size_t compressed_total = 0; };

    struct WorkItem {
        CudaUtils::unique_host_ptr<uint8_t> h_buffer;
        size_t size;
        uint64_t original_offset;
    };

    void process_tensor(const std::tuple<std::string, std::vector<int64_t>, std::string, uint64_t>& tensor_info,
                        json& telemetry_json, unsigned int stream_id) {
        const auto& [name, shape, dtype, total_bytes] = tensor_info;
        if (checkpoint_.is_done(name)) {
            LOGI("Skipping tensor '%s' (already completed)", name.c_str());
            return;
        }
        LOGI("Processing tensor '%s' (%.2f MB)...", name.c_str(), total_bytes / 1e6);

        auto t_start = std::chrono::high_resolution_clock::now();
        TensorResult result;
        result.original_bytes = total_bytes;
        
        std::vector<WorkItem> batch;
        uint64_t current_batch_bytes = 0;
        uint64_t processed_bytes = 0;

        while (processed_bytes < total_bytes && !g_cancelled) {
            size_t chunk_size = static_cast<size_t>(std::min<uint64_t>(chunk_size_, total_bytes - processed_bytes));
            auto h_buffer = CudaUtils::make_unique_host<uint8_t>(chunk_size);
            size_t bytes_read = reader_.read_tensor_chunk(name, processed_bytes, h_buffer.get(), chunk_size);
            if (bytes_read == 0) break;

            batch.emplace_back(WorkItem{std::move(h_buffer), bytes_read, processed_bytes});
            current_batch_bytes += bytes_read;
            processed_bytes += bytes_read;

            bool is_last_chunk = (processed_bytes >= total_bytes);
            if (batch.size() >= BATCH_MAX_CHUNKS || current_batch_bytes >= BATCH_MAX_BYTES_UNCOMPRESSED || is_last_chunk) {
                process_batch(batch, result, streams_[stream_id]);
                batch.clear();
                current_batch_bytes = 0;
            }
        }
        
        auto t_end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        {
            std::lock_guard<std::mutex> lock(completed_mutex_);
            completed_tensors_[name] = result;
        }
        
        std::string all_shas;
        for(const auto& c : result.chunks) all_shas += c.comp_sha256;
        std::string final_sha = CpuUtils::sha256_hex((const uint8_t*)all_shas.data(), all_shas.size());
        double ratio = total_bytes > 0 ? (double)total_bytes / result.compressed_total : 1.0;

        {
            std::lock_guard<std::mutex> lock(g_log_mutex); // Reuse log mutex for file I/O
            std::ofstream csv(TELEMETRY_CSV_FILENAME, std::ios::app);
            csv << name << "," << total_bytes << "," << result.compressed_total << "," << ratio << "," << time_ms << "," << final_sha << "\n";
            telemetry_json.push_back({
                {"name", name}, {"orig_bytes", total_bytes}, {"comp_bytes", result.compressed_total},
                {"ratio", ratio}, {"time_ms", time_ms}, {"sha", final_sha}
            });
        }
        checkpoint_.mark_done(name);
    }

    void process_batch(std::vector<WorkItem>& batch, TensorResult& result, cudaStream_t stream) {
        try {
            std::vector<CudaUtils::unique_device_ptr<uint8_t>> d_in_buffers;
            std::vector<void*> d_in_ptrs;
            std::vector<size_t> in_sizes;
            d_in_buffers.reserve(batch.size());
            d_in_ptrs.reserve(batch.size());
            in_sizes.reserve(batch.size());

            for (const auto& item : batch) {
                auto d_buffer = CudaUtils::make_unique_device<uint8_t>(item.size, stream);
                cudaMemcpyAsync(d_buffer.get(), item.h_buffer.get(), item.size, cudaMemcpyHostToDevice, stream);
                d_in_ptrs.push_back(d_buffer.get());
                in_sizes.push_back(item.size);
                d_in_buffers.push_back(std::move(d_buffer));
            }

            auto gpu_results = NvcompGpuZstd::compress_batch(d_in_ptrs, in_sizes, chunk_size_, stream);

            for (size_t i = 0; i < batch.size(); ++i) {
                const auto& comp_chunk = gpu_results.compressed_chunks[i];
                uint64_t file_offset = writer_.append_data(comp_chunk.data(), comp_chunk.size());
                result.chunks.push_back({batch[i].original_offset, batch[i].size, comp_chunk.size(), gpu_results.chunk_shas_hex[i], file_offset});
                result.compressed_total += comp_chunk.size();
            }
        } catch (const std::exception& e) {
            LOGW("GPU batch failed: %s. Falling back to CPU ZSTD for this batch.", e.what());
            for (const auto& item : batch) {
                size_t bound = ZSTD_compressBound(item.size);
                std::vector<uint8_t> compressed(bound);
                size_t csize = ZSTD_compress(compressed.data(), bound, item.h_buffer.get(), item.size, 3);
                if (ZSTD_isError(csize)) throw ConversionException("CPU ZSTD fallback failed");
                compressed.resize(csize);
                std::string sha = CpuUtils::sha256_hex(compressed.data(), csize);
                uint64_t file_offset = writer_.append_data(compressed.data(), csize);
                result.chunks.push_back({item.original_offset, item.size, csize, sha, file_offset});
                result.compressed_total += csize;
            }
        }
    }

    ggml_type parse_dtype(const std::string& s) {
        static const std::unordered_map<std::string, ggml_type> m = {
            {"F32", GGML_TYPE_F32}, {"F16", GGML_TYPE_F16}, {"BF16", GGML_TYPE_BF16},
            {"I32", GGML_TYPE_I32}, {"I16", GGML_TYPE_I16}, {"I8", GGML_TYPE_I8}, {"U8", GGML_TYPE_U8}
        };
        auto it = m.find(s);
        return (it != m.end()) ? it->second : GGML_TYPE_F32;
    }

    SafeTensorsStreamReader reader_;
    GGUFWriter writer_;
    size_t chunk_size_;
    Checkpoint checkpoint_;
    std::vector<cudaStream_t> streams_;
    std::mutex completed_mutex_;
    std::unordered_map<std::string, TensorResult> completed_tensors_;
};

// ----------------------------------------------------------------------------
// Main Entry Point
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.safetensors> <output.gguf> [chunk_bytes] [cuda_streams] [threads]\n", argv[0]);
        return 1;
    }
#ifdef _WIN32
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
#else
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = signal_handler;
    sigaction(SIGINT, &action, NULL);
    sigaction(SIGTERM, &action, NULL);
#endif

    try {
        std::string input = argv[1];
        std::string output = argv[2];
        size_t chunk = (argc >= 4) ? std::stoull(argv[3]) : CHUNK_DEFAULT_SIZE_BYTES;
        chunk = std::clamp(chunk, CHUNK_MIN_SIZE_BYTES, CHUNK_MAX_SIZE_BYTES);
        
        unsigned int streams = (argc >= 5) ? std::stoul(argv[4]) : 4;
        streams = std::max(1u, streams);

        unsigned int threads = (argc >= 6) ? std::stoul(argv[5]) : std::thread::hardware_concurrency();
        threads = std::max(1u, threads);

        LOGI("AdvancedGGUF_Converter v7.0 starting...");
        LOGI("Input: %s, Output: %s", input.c_str(), output.c_str());
        LOGI("Config: ChunkSize=%.1fMB, CudaStreams=%u, WorkerThreads=%u", chunk / 1e6, streams, threads);
        
        Converter conv(input, output, chunk, streams);
        int rc = conv.run(threads);

        if (g_cancelled) {
            LOGW("Conversion was cancelled by the user.");
            return 2;
        }
        
        LOGI("Conversion finished with exit code %d.", rc);
        return rc;

    } catch (const std::exception& e) {
        LOGE("A fatal error occurred: %s", e.what());
        return 1;
    }
}
'@
}
