function Get-EnhancedConverterSource {
    return @'
#include <cstdio>
#include <cstdlib>
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

#define GGML_BUILD
#include "ggml.h"
#include "ggml-cuda.h"
#include "llama.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#include <signal.h>
#endif

// Enhanced JSON parsing with better error handling
#define JSON_DIAGNOSTICS 1
#include "json.hpp"
using json = nlohmann::json;

// DFloat11 enhanced interface
extern "C" {
    size_t DFloat11_compress_bound(size_t size);
    int    DFloat11_compress(const uint8_t* src, size_t src_size, uint8_t* dst, size_t* dst_size);
    int    DFloat11_decompress(const uint8_t* src, size_t src_size, uint8_t* dst, size_t dst_size);
}

// Enhanced ZSTD interface
#include <zstd.h>
#ifdef _MSC_VER
#pragma comment(lib, "zstd.lib")
#pragma comment(lib, "dfloat11.lib")
#pragma comment(lib, "ggml.lib")
#pragma comment(lib, "llama.lib")
#endif

// Enhanced tensor structure with metadata
struct EnhancedTensor {
    std::string name;
    ggml_type   original_type;
    ggml_type   storage_type;
    std::vector<int64_t> shape;
    std::vector<uint8_t> original_data;
    std::vector<uint8_t> compressed_data;
    
    // Compression metadata
    enum CompressionType { NONE, ZSTD, DFLOAT11 };
    CompressionType compression_type = NONE;
    int compression_level = 3;
    
    // Statistics
    size_t original_bytes = 0;
    size_t compressed_bytes = 0;
    double compression_ratio = 1.0;
    std::chrono::milliseconds compression_time{0};
    
    // Memory management
    bool is_loaded = false;
    bool is_compressed = false;
    
    void clear_original() {
        original_data.clear();
        original_data.shrink_to_fit();
        is_loaded = false;
    }
    
    double get_compression_ratio() const {
        return original_bytes > 0 ? static_cast<double>(original_bytes) / compressed_bytes : 1.0;
    }
};

// Global state with better thread safety
namespace GlobalState {
    std::atomic<size_t> tensors_processed{0};
    std::atomic<size_t> tensors_total{0};
    std::atomic<size_t> bytes_processed{0};
    std::atomic<size_t> bytes_total{0};
    std::atomic<bool> cancellation_requested{false};
    std::mutex print_mutex;
    std::mutex stats_mutex;
    
    struct Statistics {
        size_t total_tensors = 0;
        size_t compressed_tensors = 0;
        size_t dfloat11_tensors = 0;
        size_t zstd_tensors = 0;
        size_t uncompressed_tensors = 0;
        size_t total_original_bytes = 0;
        size_t total_compressed_bytes = 0;
        std::chrono::milliseconds total_time{0};
    } stats;
}

// Enhanced logging with levels
enum LogLevel { DEBUG, INFO, WARNING, ERROR };

static void log_message(LogLevel level, const char* fmt, ...) {
    std::lock_guard<std::mutex> lock(GlobalState::print_mutex);
    
    const char* level_str[] = {"DEBUG", "INFO", "WARN", "ERROR"};
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    char timestamp[100];
    std::strftime(timestamp, sizeof(timestamp), "%H:%M:%S", std::localtime(&time_t));
    
    printf("[%s.%03d] [%s] ", timestamp, (int)ms.count(), level_str[level]);
    
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    
    printf("\n");
    fflush(stdout);
}

#define LOG_DEBUG(fmt, ...) log_message(DEBUG, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  log_message(INFO, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  log_message(WARNING, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) log_message(ERROR, fmt, ##__VA_ARGS__)

// Enhanced signal handling
#ifdef _WIN32
BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT) {
        LOG_INFO("Cancellation requested by user");
        GlobalState::cancellation_requested = true;
        return TRUE;
    }
    return FALSE;
}
#else
void signal_handler(int signal) {
    LOG_INFO("Cancellation requested by signal %d", signal);
    GlobalState::cancellation_requested = true;
}
#endif

// Enhanced SafeTensors loader with validation
class SafeTensorsLoader {
public:
    static std::vector<EnhancedTensor> load(const std::filesystem::path& path) {
        LOG_INFO("Loading SafeTensors file: %s", path.string().c_str());
        
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + path.string());
        }
        
        // Read header length
        uint64_t header_len = 0;
        file.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
        
        if (header_len == 0 || header_len > 100 * 1024 * 1024) { // Sanity check: max 100MB header
            throw std::runtime_error("Invalid header length: " + std::to_string(header_len));
        }
        
        // Read and parse header
        std::vector<char> header_data(header_len);
        file.read(header_data.data(), header_len);
        
        std::string header_str(header_data.begin(), header_data.end());
        json header_json;
        
        try {
            header_json = json::parse(header_str);
        } catch (const json::exception& e) {
            throw std::runtime_error("Failed to parse JSON header: " + std::string(e.what()));
        }
        
        std::vector<EnhancedTensor> tensors;
        size_t total_size = 0;
        
        for (auto& [name, meta] : header_json.items()) {
            if (name == "__metadata__") continue; // Skip metadata
            
            EnhancedTensor tensor;
            tensor.name = name;
            
            // Parse shape
            if (!meta.contains("shape") || !meta["shape"].is_array()) {
                throw std::runtime_error("Invalid shape for tensor: " + name);
            }
            for (auto& dim : meta["shape"]) {
                tensor.shape.push_back(dim.get<int64_t>());
            }
            
            // Parse data type
            std::string dtype = meta.value("dtype", "unknown");
            tensor.original_type = parse_dtype(dtype);
            tensor.storage_type = tensor.original_type;
            
            // Parse data offsets
            if (!meta.contains("data_offsets") || !meta["data_offsets"].is_array() || 
                meta["data_offsets"].size() != 2) {
                throw std::runtime_error("Invalid data_offsets for tensor: " + name);
            }
            
            uint64_t start_offset = meta["data_offsets"][0];
            uint64_t end_offset = meta["data_offsets"][1];
            
            if (end_offset <= start_offset) {
                throw std::runtime_error("Invalid data range for tensor: " + name);
            }
            
            tensor.original_bytes = end_offset - start_offset;
            total_size += tensor.original_bytes;
            
            // Load tensor data
            tensor.original_data.resize(tensor.original_bytes);
            file.seekg(8 + header_len + start_offset);
            file.read(reinterpret_cast<char*>(tensor.original_data.data()), tensor.original_bytes);
            
            if (file.gcount() != tensor.original_bytes) {
                throw std::runtime_error("Failed to read tensor data for: " + name);
            }
            
            tensor.is_loaded = true;
            tensors.push_back(std::move(tensor));
        }
        
        LOG_INFO("Loaded %zu tensors, total size: %.2f MB", 
                 tensors.size(), total_size / 1e6);
        
        GlobalState::tensors_total = tensors.size();
        GlobalState::bytes_total = total_size;
        
        return tensors;
    }
    
private:
    static ggml_type parse_dtype(const std::string& dtype) {
        static const std::unordered_map<std::string, ggml_type> type_map = {
            {"F32", GGML_TYPE_F32}, {"float32", GGML_TYPE_F32},
            {"F16", GGML_TYPE_F16}, {"float16", GGML_TYPE_F16},
            {"BF16", GGML_TYPE_BF16}, {"bfloat16", GGML_TYPE_BF16},
            {"I32", GGML_TYPE_I32}, {"int32", GGML_TYPE_I32},
            {"I16", GGML_TYPE_I16}, {"int16", GGML_TYPE_I16},
            {"I8", GGML_TYPE_I8}, {"int8", GGML_TYPE_I8},
            {"U8", GGML_TYPE_I8}, {"uint8", GGML_TYPE_I8}
        };
        
        auto it = type_map.find(dtype);
        if (it != type_map.end()) {
            return it->second;
        }
        
        throw std::runtime_error("Unsupported data type: " + dtype);
    }
};

// Enhanced compression strategy
class CompressionStrategy {
public:
    static EnhancedTensor::CompressionType select_compression(const EnhancedTensor& tensor) {
        // Strategy based on tensor characteristics
        bool is_weight = tensor.name.find("weight") != std::string::npos;
        bool is_moe = tensor.name.find("feed_forward") != std::string::npos || 
                     tensor.name.find("expert") != std::string::npos;
        bool is_embedding = tensor.name.find("embed") != std::string::npos;
        
        // Use DFloat11 for MoE weights and large embedding layers
        if ((is_moe || is_embedding) && 
            (tensor.original_type == GGML_TYPE_F16 || tensor.original_type == GGML_TYPE_F32) &&
            tensor.original_bytes > 1024 * 1024) { // > 1MB
            return EnhancedTensor::DFLOAT11;
        }
        
        // Use ZSTD for other data
        if (tensor.original_bytes > 4096) { // > 4KB
            return EnhancedTensor::ZSTD;
        }
        
        // No compression for small tensors
        return EnhancedTensor::NONE;
    }
    
    static bool compress_tensor(EnhancedTensor& tensor) {
        if (GlobalState::cancellation_requested) {
            return false;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        tensor.compression_type = select_compression(tensor);
        
        bool success = false;
        
        switch (tensor.compression_type) {
            case EnhancedTensor::DFLOAT11:
                success = compress_with_dfloat11(tensor);
                break;
            case EnhancedTensor::ZSTD:
                success = compress_with_zstd(tensor);
                break;
            case EnhancedTensor::NONE:
                success = copy_uncompressed(tensor);
                break;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        tensor.compression_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        if (success) {
            tensor.compression_ratio = tensor.get_compression_ratio();
            tensor.is_compressed = true;
            
            // Update global statistics
            GlobalState::tensors_processed++;
            GlobalState::bytes_processed += tensor.original_bytes;
            
            // Progress reporting
            double progress = static_cast<double>(GlobalState::tensors_processed) / 
                            GlobalState::tensors_total * 100.0;
            
            LOG_INFO("[%.1f%%] %s: %.2f→%.2f MB (%.2fx) in %dms", 
                     progress, tensor.name.c_str(),
                     tensor.original_bytes / 1e6, tensor.compressed_bytes / 1e6,
                     tensor.compression_ratio, (int)tensor.compression_time.count());
        }
        
        return success;
    }
    
private:
    static bool compress_with_dfloat11(EnhancedTensor& tensor) {
        size_t bound = DFloat11_compress_bound(tensor.original_bytes);
        tensor.compressed_data.resize(bound);
        
        size_t compressed_size = bound;
        int result = DFloat11_compress(
            tensor.original_data.data(), tensor.original_bytes,
            tensor.compressed_data.data(), &compressed_size
        );
        
        if (result == 0) {
            tensor.compressed_data.resize(compressed_size);
            tensor.compressed_bytes = compressed_size;
            return true;
        }
        
        LOG_WARN("DFloat11 compression failed for %s, falling back to ZSTD", 
                 tensor.name.c_str());
        return compress_with_zstd(tensor);
    }
    
    static bool compress_with_zstd(EnhancedTensor& tensor) {
        int level = std::getenv("COMPRESSION_LEVEL") ? 
                   std::atoi(std::getenv("COMPRESSION_LEVEL")) : 3;
        
        size_t bound = ZSTD_compressBound(tensor.original_bytes);
        tensor.compressed_data.resize(bound);
        
        size_t compressed_size = ZSTD_compress(
            tensor.compressed_data.data(), bound,
            tensor.original_data.data(), tensor.original_bytes,
            level
        );
        
        if (!ZSTD_isError(compressed_size)) {
            tensor.compressed_data.resize(compressed_size);
            tensor.compressed_bytes = compressed_size;
            tensor.compression_type = EnhancedTensor::ZSTD;
            return true;
        }
        
        LOG_ERROR("ZSTD compression failed for %s: %s", 
                  tensor.name.c_str(), ZSTD_getErrorName(compressed_size));
        return copy_uncompressed(tensor);
    }
    
    static bool copy_uncompressed(EnhancedTensor& tensor) {
        tensor.compressed_data = tensor.original_data;
        tensor.compressed_bytes = tensor.original_bytes;
        tensor.compression_type = EnhancedTensor::NONE;
        return true;
    }
};

// Enhanced GGUF writer with metadata
class GGUFWriter {
public:
    static void write(const std::filesystem::path& output_path, 
                     std::vector<EnhancedTensor>& tensors) {
        LOG_INFO("Writing GGUF file: %s", output_path.string().c_str());
        
        struct gguf_context* ctx = gguf_init_empty();
        if (!ctx) {
            throw std::runtime_error("Failed to initialize GGUF context");
        }
        
        try {
            // Set metadata
            gguf_set_val_str(ctx, "general.name", "AdvancedCompressed");
            gguf_set_val_str(ctx, "general.architecture", "llama");
            gguf_set_val_str(ctx, "general.version", "2.0");
            gguf_set_val_str(ctx, "compression.method", "DFloat11+ZSTD");
            gguf_set_val_u32(ctx, "compression.level", 3);
            gguf_set_val_str(ctx, "compression.created_by", "AdvancedGGUF_Converter v2.0");
            
            // Add compression statistics
            size_t dfloat11_count = 0, zstd_count = 0, uncompressed_count = 0;
            for (const auto& tensor : tensors) {
                switch (tensor.compression_type) {
                    case EnhancedTensor::DFLOAT11: dfloat11_count++; break;
                    case EnhancedTensor::ZSTD: zstd_count++; break;
                    case EnhancedTensor::NONE: uncompressed_count++; break;
                }
            }
            
            gguf_set_val_u32(ctx, "compression.dfloat11_tensors", dfloat11_count);
            gguf_set_val_u32(ctx, "compression.zstd_tensors", zstd_count);
            gguf_set_val_u32(ctx, "compression.uncompressed_tensors", uncompressed_count);
            
            // Add tensor definitions
            for (const auto& tensor : tensors) {
                gguf_add_tensor(ctx, tensor.name.c_str(), 
                              tensor.shape.data(), tensor.shape.size(),
                              tensor.storage_type, nullptr);
            }
            
            // Write header
            FILE* file = fopen(output_path.string().c_str(), "wb");
            if (!file) {
                throw std::runtime_error("Cannot create output file");
            }
            
            gguf_write_to_file(ctx, output_path.string().c_str());
            
            // Write tensor data
            for (const auto& tensor : tensors) {
                size_t written = fwrite(tensor.compressed_data.data(), 1, 
                                      tensor.compressed_bytes, file);
                if (written != tensor.compressed_bytes) {
                    fclose(file);
                    throw std::runtime_error("Failed to write tensor data");
                }
            }
            
            fclose(file);
            
        } catch (...) {
            gguf_free(ctx);
            throw;
        }
        
        gguf_free(ctx);
        LOG_INFO("GGUF file written successfully");
    }
};

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.safetensors> <output.gguf>\n", argv[0]);
        return 1;
    }
    
    // Setup signal handling
#ifdef _WIN32
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
#else
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
#endif
    
    try {
        LOG_INFO("Advanced GGUF Converter v2.0 starting");
        LOG_INFO("Input: %s", argv[1]);
        LOG_INFO("Output: %s", argv[2]);
        
        // Load tensors
        auto tensors = SafeTensorsLoader::load(argv[1]);
        
        if (GlobalState::cancellation_requested) {
            LOG_INFO("Cancelled by user");
            return 1;
        }
        
        // Setup threading
        unsigned int thread_count = std::thread::hardware_concurrency();
        if (const char* env_threads = std::getenv("OMP_NUM_THREADS")) {
            thread_count = std::max(1u, std::min(thread_count, 
                                                 (unsigned)std::atoi(env_threads)));
        }
        
        LOG_INFO("Compressing %zu tensors using %u threads", 
                 tensors.size(), thread_count);
        
        // Compress tensors in parallel
        std::vector<std::thread> workers;
        std::atomic<size_t> tensor_index{0};
        
        for (unsigned int i = 0; i < thread_count; ++i) {
            workers.emplace_back([&]() {
                while (!GlobalState::cancellation_requested) {
                    size_t idx = tensor_index.fetch_add(1);
                    if (idx >= tensors.size()) break;
                    
                    if (!CompressionStrategy::compress_tensor(tensors[idx])) {
                        LOG_ERROR("Failed to compress tensor: %s", 
                                  tensors[idx].name.c_str());
                    }
                }
            });
        }
        
        // Wait for completion
        for (auto& worker : workers) {
            worker.join();
        }
        
        if (GlobalState::cancellation_requested) {
            LOG_INFO("Cancelled by user");
            return 1;
        }
        
        // Calculate final statistics
        size_t total_original = 0, total_compressed = 0;
        for (const auto& tensor : tensors) {
            total_original += tensor.original_bytes;
            total_compressed += tensor.compressed_bytes;
        }
        
        double overall_ratio = total_original > 0 ? 
                              static_cast<double>(total_original) / total_compressed : 1.0;
        
        LOG_INFO("Compression complete: %.2f GB → %.2f GB (%.2fx reduction)", 
                 total_original / 1e9, total_compressed / 1e9, overall_ratio);
        
        // Write output
        GGUFWriter::write(argv[2], tensors);
        
        LOG_INFO("Conversion completed successfully!");
        return 0;
        
    } catch (const std::exception& e) {
        LOG_ERROR("FATAL: %s", e.what());
        return 1;
    }
}
'@
}
