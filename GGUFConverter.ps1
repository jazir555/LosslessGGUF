#Requires -Version 5.1
# ---
# Advanced LOSSLESS GGUF Converter
# ---
# This script achieves maximum lossless compression by using model-aware strategies.
# The output is a specialized GGUF requiring a custom runtime for 1:1 reconstruction.
#
# Core Lossless Tactics:
# 1. Delta Compression: Stores the exact difference between similar layers.
# 2. Learned Dictionary: Replaces common weight patterns with short codes.
# 3. Heuristic Algorithm Selection: Uses Brotli/LZ4 based on tensor type.
# ---
Add-Type -AssemblyName System.Windows.Forms
param(
    [string]$ModelPath,
    [switch]$SelfUpdate,
    [switch]$Pareto,
    [int]$GpuLayers = 999
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$form = New-Object System.Windows.Forms.Form
$form.Text        = 'Advanced Lossless GGUF Converter (100% Quality)'
$form.Size        = New-Object System.Drawing.Size(900,700)
$form.StartPosition = 'CenterScreen'
$form.MinimumSize = $form.Size

$txtLog           = New-Object System.Windows.Forms.TextBox
$txtLog.Dock      = 'Fill'
$txtLog.Multiline = $true
$txtLog.ScrollBars= 'Vertical'
$txtLog.Font      = New-Object System.Drawing.Font('Consolas',9)
$txtLog.ReadOnly  = $true
$form.Controls.Add($txtLog)

function Log {
    param([string]$line)
    $stamp = (Get-Date -Format 'HH:mm:ss.fff')
    $txtLog.AppendText("[$stamp]  $line`r`n")
    $txtLog.SelectionStart = $txtLog.Text.Length
    $txtLog.ScrollToCaret()
}

$scriptDir   = $PSScriptRoot
$toolsDir    = Join-Path $scriptDir '_tools'
$outDir      = Join-Path $scriptDir '_out'
$venvsDir    = Join-Path $scriptDir '_venvs'
$srcDir      = Join-Path $toolsDir 'src'
$buildDir    = Join-Path $toolsDir 'build'
$binDir      = Join-Path $toolsDir 'bin'

@($toolsDir,$outDir,$venvsDir,$srcDir,$buildDir,$binDir) |
    Where-Object { -not (Test-Path $_) } |
    ForEach-Object { New-Item -ItemType Directory -Path $_ -Force | Out-Null }

$pyExe = Join-Path $venvsDir 'converter\Scripts\python.exe'
if (-not (Test-Path $pyExe)) {
    Log 'Creating Python 3.11 venv…'
    $embedZip = Join-Path $toolsDir 'python-3.11.9-embed-amd64.zip'
    if (-not (Test-Path $embedZip)) {
        Log 'Downloading embedded Python…'
        Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip' -OutFile $embedZip
    }
    Expand-Archive -Path $embedZip -DestinationPath (Join-Path $venvsDir 'converter') -Force
    $pipPy = Join-Path $toolsDir 'get-pip.py'
    if (-not (Test-Path $pipPy)) {
        Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile $pipPy
    }
    $pyExe = Join-Path $venvsDir 'converter\python.exe'
    & $pyExe $pipPy 2>$null
    if ($LASTEXITCODE -ne 0) {
        Log 'Embedded python pip setup failed. Using system Python to create venv…'
        try {
            $pyExe = (Get-Command python -ErrorAction Stop).Source
            & $pyExe -m venv (Join-Path $venvsDir 'converter')
            $pyExe = Join-Path $venvsDir 'converter\Scripts\python.exe'
        } catch {
            throw 'Python is required to create a venv. Please install Python 3.10+.'
        }
    }
}
& $pyExe -m pip install -q -U pip wheel cmake ninja

function Find-Msvc {
    $vw = Join-Path ${env:ProgramFiles(x88)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path $vw) {
        $path = & $vw -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($path) {
            $bat = Join-Path $path 'VC\Auxiliary\Build\vcvars64.bat'
            if (Test-Path $bat) { return $bat }
        }
    }
    throw 'MSVC not found. Install Visual Studio Build Tools 2022 (C++ workload) and retry.'
}
$vcvars = Find-Msvc

if ($SelfUpdate) {
    Log 'Self-update requested…'
    @('llama.cpp', 'lz4', 'brotli') | ForEach-Object {
        Remove-Item -Recurse -Force (Join-Path $srcDir $_) -ErrorAction SilentlyContinue
    }
}

function Clone-Repo {
    param($url,$folder)
    $target = Join-Path $srcDir $folder
    if (-not (Test-Path $target)) {
        Log "Cloning $folder…"
        & git clone --depth 1 --quiet $url $target
    } else {
        Log "Updating $folder…"
        Push-Location $target
        & git fetch --depth 1 origin
        & git reset --hard origin/HEAD
        Pop-Location
    }
}
Clone-Repo 'https://github.com/ggerganov/llama.cpp.git' 'llama.cpp'
Clone-Repo 'https://github.com/lz4/lz4.git' 'lz4'
Clone-Repo 'https://github.com/google/brotli.git' 'brotli'

$lz4Lib = Join-Path $buildDir 'lz4.lib'
Push-Location (Join-Path $srcDir 'lz4\build\cmake')
if (-not (Test-Path $lz4Lib)) {
    Log 'Building lz4...'
    & cmake -B (Join-Path $buildDir 'lz4') -G Ninja "-DCMAKE_BUILD_TYPE=Release" "-DBUILD_SHARED_LIBS=OFF"
    & cmake --build (Join-Path $buildDir 'lz4') --target lz4_static
    Copy-Item (Join-Path $buildDir 'lz4\lib\lz4_static.lib') $lz4Lib
}
Pop-Location

$brotliEncLib = Join-Path $buildDir 'brotlienc.lib'
Push-Location (Join-Path $srcDir 'brotli')
if (-not (Test-Path $brotliEncLib)) {
    Log 'Building brotli...'
    & cmake -B (Join-Path $buildDir 'brotli') -G Ninja "-DCMAKE_BUILD_TYPE=Release" "-DBUILD_SHARED_LIBS=OFF"
    & cmake --build (Join-Path $buildDir 'brotli') --target brotlienc-static brotlidec-static brotlicommon-static
    Copy-Item (Join-Path $buildDir 'brotli\brotlienc-static.lib') $brotliEncLib
    Copy-Item (Join-Path $buildDir 'brotli\brotlidec-static.lib') (Join-Path $buildDir 'brotlidec.lib')
    Copy-Item (Join-Path $buildDir 'brotli\brotlicommon-static.lib') (Join-Path $buildDir 'brotlicommon.lib')
}
Pop-Location

$llamaSrc = Join-Path $srcDir 'llama.cpp'
$ggmlLib  = Join-Path $buildDir 'ggml.lib'
if (-not (Test-Path $ggmlLib)) {
    Log 'Building ggml + llama (static)…'
    Push-Location $llamaSrc
    $defs = @(
        '-DCMAKE_BUILD_TYPE=Release',
        '-DBUILD_SHARED_LIBS=OFF',
        '-DLLAMA_BUILD_TESTS=OFF',
        '-DLLAMA_BUILD_EXAMPLES=OFF',
        '-DLLAMA_CUDA=OFF', # No GPU needed for conversion logic
        '-DCMAKE_CXX_FLAGS="-march=native"'
    )
    & cmake -B $buildDir -G Ninja $defs
    & cmake --build $buildDir --target ggml llama
    Pop-Location
}

$converterCpp = Join-Path $toolsDir 'lossless_converter.cpp'
$converterExe = Join-Path $binDir   'lossless_converter.exe'

# --- C++ Source Code ---
$code = @'
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
#include <regex>
#include <map>

#define GGML_BUILD
#include "ggml.h"
#include "gguf.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#include <sys/mman.h>
#endif

#include "json.hpp"
using json = nlohmann::json;

#include <lz4.h>
#include <brotli/encode.h>

#ifdef _MSC_VER
#pragma comment(lib,"lz4.lib")
#pragma comment(lib,"ggml.lib")
#pragma comment(lib,"brotlienc.lib")
#pragma comment(lib,"brotlicommon.lib")
#endif

enum class CompressionType {
    NONE,
    LZ4,
    BROTLI,
    LEARNED, // Dictionary-based on byte patterns
    PARETO   // Optimal LZ4 level finding
};

struct Tensor {
    std::string name;
    ggml_type   dtype;
    std::vector<int64_t> shape;
    const uint8_t* data_ptr = nullptr;

    std::vector<uint8_t> comp;
    CompressionType comp_type = CompressionType::NONE;
    bool is_delta = false;
    
    size_t orig_bytes = 0;
    size_t comp_bytes = 0;
};

namespace globals {
    std::atomic<size_t> processed{0};
    std::atomic<size_t> total{0};
    std::mutex print_mux;
    std::unordered_map<std::string, const Tensor*> previous_tensors;
    std::map<std::vector<uint8_t>, uint16_t> dictionary;
    uint16_t next_dict_code = 0;
}

static void print(const char* fmt, ...) {
    std::lock_guard<std::mutex> lock(globals::print_mux);
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    fflush(stdout);
}

struct MappedFile {
#ifdef _WIN32
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMapping = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
    const uint8_t* data = nullptr;
    uint64_t size = 0;

    MappedFile(const std::filesystem::path& path) {
#ifdef _WIN32
        hFile = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open file");
        LARGE_INTEGER li;
        if (!GetFileSizeEx(hFile, &li)) { CloseHandle(hFile); throw std::runtime_error("cannot get file size"); }
        size = li.QuadPart;
        hMapping = CreateFileMapping(hFile, NULL, PAGE_READONLY | SEC_LARGE_PAGES, 0, 0, NULL);
        if (hMapping == NULL) {
            print("Could not create file mapping with huge pages, falling back. For huge pages, grant 'SeLockMemoryPrivilege' to the user.\n");
            hMapping = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        }
        if (hMapping == NULL) { CloseHandle(hFile); throw std::runtime_error("cannot create file mapping"); }
        data = (const uint8_t*)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        if (data == nullptr) { CloseHandle(hMapping); CloseHandle(hFile); throw std::runtime_error("cannot map view of file"); }
#else
        fd = open(path.c_str(), O_RDONLY);
        if (fd == -1) throw std::runtime_error("cannot open file");
        struct stat st;
        if (fstat(fd, &st) == -1) { close(fd); throw std::runtime_error("cannot get file size"); }
        size = st.st_size;
        data = (const uint8_t*)mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (data == MAP_FAILED) { close(fd); throw std::runtime_error("cannot map file"); }
#endif
    }
    ~MappedFile() {
#ifdef _WIN32
        if (data) UnmapViewOfFile(data);
        if (hMapping) CloseHandle(hMapping);
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
#else
        if (data) munmap((void*)data, size);
        if (fd != -1) close(fd);
#endif
    }
};

static void build_dictionary(const std::vector<Tensor>& tensors) {
    std::map<std::vector<uint8_t>, int> counts;
    for (const auto& t : tensors) {
        if (t.dtype == GGML_TYPE_F16) {
            for (size_t i = 0; i + 4 <= t.orig_bytes; i += 4) {
                std::vector<uint8_t> seq(t.data_ptr + i, t.data_ptr + i + 4);
                counts[seq]++;
            }
        }
    }
    std::vector<std::pair<std::vector<uint8_t>, int>> sorted_counts(counts.begin(), counts.end());
    std::sort(sorted_counts.begin(), sorted_counts.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    
    globals::dictionary.clear();
    globals::next_dict_code = 0;
    for (int i = 0; i < std::min((int)sorted_counts.size(), 65534); ++i) { // Reserve 0xFFFF for escape
        globals::dictionary[sorted_counts[i].first] = globals::next_dict_code++;
    }
}

static std::vector<Tensor> load_safe_tensors(const MappedFile& file) {
    const uint8_t* ptr = file.data;
    uint64_t header_len = *(uint64_t*)ptr;
    ptr += 8;
    if (header_len == 0 || header_len > file.size) throw std::runtime_error("invalid safetensors header");
    std::string json_str((char*)ptr, header_len);
    const uint8_t* data_start = file.data + 8 + header_len;

    auto j = json::parse(json_str);
    std::vector<Tensor> tensors;
    for (auto& [name, meta] : j.items()) {
        Tensor t;
        t.name = name;
        for (auto& s : meta["shape"]) t.shape.push_back(s.get<int64_t>());
        std::string dtype = meta["dtype"];
        if (dtype == "F32") t.dtype = GGML_TYPE_F32;
        else if (dtype == "F16") t.dtype = GGML_TYPE_F16;
        else if (dtype == "BF16") t.dtype = GGML_TYPE_BF16;
        else continue;
        uint64_t offset = meta["data_offsets"][0];
        size_t   sz     = meta["data_offsets"][1] - offset;
        t.orig_bytes = sz;
        t.data_ptr = data_start + offset;
        tensors.emplace_back(std::move(t));
    }
    globals::total = tensors.size();
    return tensors;
}

static void compress_tensor(Tensor& t, bool use_pareto) {
    bool is_embedding_or_output = (t.name.find("embed_tokens") != std::string::npos || t.name.find("lm_head") != std::string::npos);

    // Step 1: Delta Compression (Lossless)
    const uint8_t* data_to_compress = t.data_ptr;
    std::vector<uint8_t> delta_data;
    if (!is_embedding_or_output && t.name.find("layers.") != std::string::npos) {
        std::string base_name = std::regex_replace(t.name, std::regex(R"(\.layers\.\d+\.)"), ".layers.0.");
        auto it = globals::previous_tensors.find(base_name);
        if (it != globals::previous_tensors.end()) {
            const Tensor* prev_t = it->second;
            if (prev_t->orig_bytes == t.orig_bytes) {
                t.is_delta = true;
                delta_data.resize(t.orig_bytes);
                for (size_t i = 0; i < t.orig_bytes; ++i) {
                    delta_data[i] = t.data_ptr[i] - prev_t->data_ptr[i];
                }
                data_to_compress = delta_data.data();
            }
        }
        // Store a pointer to the current tensor for the next layer's delta calculation
        globals::previous_tensors[std::regex_replace(t.name, std::regex(R"(\.layers\.\d+\.)"), ".layers.0.")] = &t;
    }

    // Step 2: Choose primary lossless algorithm
    if (use_pareto) {
        t.comp_type = CompressionType::PARETO;
        double best_score = -1.0;
        int best_level = 1;
        for (int level = 1; level <= 12; ++level) {
            std::vector<uint8_t> temp_comp(LZ4_compressBound(t.orig_bytes));
            const int comp_len = LZ4_compress_fast((const char*)data_to_compress, (char*)temp_comp.data(), t.orig_bytes, temp_comp.size(), level);
            if (comp_len <= 0) continue;
            double ratio = (double)t.orig_bytes / comp_len;
            double score = ratio - (level * 0.1); // Simple score: favor ratio, penalize high effort
            if (score > best_score) {
                best_score = score;
                best_level = level;
            }
        }
        t.comp.resize(LZ4_compressBound(t.orig_bytes));
        const int comp_len = LZ4_compress_fast((const char*)data_to_compress, (char*)t.comp.data(), t.orig_bytes, t.comp.size(), best_level);
        t.comp.resize(comp_len);

    } else if (t.dtype == GGML_TYPE_F16 && !globals::dictionary.empty()) {
        t.comp_type = CompressionType::LEARNED;
        for (size_t i = 0; i < t.orig_bytes; ) {
            if (i + 4 <= t.orig_bytes) {
                std::vector<uint8_t> seq(data_to_compress + i, data_to_compress + i + 4);
                auto it = globals::dictionary.find(seq);
                if (it != globals::dictionary.end()) {
                    uint16_t code = it->second;
                    t.comp.push_back(code & 0xFF);
                    t.comp.push_back(code >> 8);
                    i += 4;
                    continue;
                }
            }
            // Escape code (0xFFFF) followed by a literal byte
            t.comp.push_back(0xFF); t.comp.push_back(0xFF);
            t.comp.push_back(data_to_compress[i]);
            i++;
        }
    } else if (is_embedding_or_output) {
        t.comp_type = CompressionType::BROTLI;
        t.comp.resize(BrotliEncoderMaxCompressedSize(t.orig_bytes));
        size_t comp_len = t.comp.size();
        BrotliEncoderCompress(BROTLI_DEFAULT_QUALITY, BROTLI_DEFAULT_WINDOW, BROTLI_MODE_GENERIC, t.orig_bytes, data_to_compress, &comp_len, t.comp.data());
        t.comp.resize(comp_len);
    } else {
        t.comp_type = CompressionType::LZ4;
        t.comp.resize(LZ4_compressBound(t.orig_bytes));
        const int comp_len = LZ4_compress_default((const char*)data_to_compress, (char*)t.comp.data(), t.orig_bytes, t.comp.size());
        t.comp.resize(comp_len > 0 ? comp_len : 0);
    }

    t.comp_bytes = t.comp.size();
    globals::processed++;
    print("\r[%zu/%zu] %-55s (delta:%d) %.2f->%.2f MB (%.2f:1)",
          globals::processed.load(), globals::total.load(),
          t.name.c_str(), t.is_delta, t.orig_bytes / 1e6, t.comp_bytes / 1e6,
          (t.comp_bytes > 0) ? double(t.orig_bytes) / t.comp_bytes : 0);
}

static void write_gguf(const std::filesystem::path& out_path, std::vector<Tensor>& tensors) {
    gguf_context* ctx = gguf_init_empty();
    gguf_set_val_str(ctx, "general.name", "LosslessCompressedModel");
    gguf_set_val_str(ctx, "general.architecture", "llama");

    if (!globals::dictionary.empty()) {
        std::vector<uint8_t> dict_data;
        std::vector<std::vector<uint8_t>> code_to_seq(globals::next_dict_code);
        for(const auto& [seq, code] : globals::dictionary) code_to_seq[code] = seq;
        for(const auto& seq : code_to_seq) dict_data.insert(dict_data.end(), seq.begin(), seq.end());
        gguf_add_tensor(ctx, "lossless.dictionary", { (int64_t)globals::next_dict_code, 4 }, 2, GGML_TYPE_U8, dict_data.data());
    }

    for (const auto& t : tensors) {
        gguf_add_tensor(ctx, t.name.c_str(), t.shape.data(), t.shape.size(), t.dtype, nullptr);
        const char* comp_str = "NONE";
        switch (t.comp_type) {
            case CompressionType::LZ4: comp_str = "LZ4"; break;
            case CompressionType::BROTLI: comp_str = "BROTLI"; break;
            case CompressionType::LEARNED: comp_str = "LEARNED"; break;
            case CompressionType::PARETO: comp_str = "PARETO_LZ4"; break;
        }
        gguf_set_tensor_str(ctx, t.name.c_str(), "lossless.compression", comp_str);
        if (t.is_delta) gguf_set_tensor_val_bool(ctx, t.name.c_str(), "lossless.is_delta", true);
    }

    FILE* f = std::fopen(out_path.string().c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open output");
    
    // Write header and KV data first
    gguf_write_to_file(ctx, f, false);

    // Now append all tensor data
    for (const auto& t : tensors) {
        std::fwrite(t.comp.data(), 1, t.comp.size(), f);
    }
    std::fclose(f);
    gguf_free(ctx);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.safetensors out.gguf [-Pareto]\n", argv[0]);
        return 1;
    }
    bool use_pareto = (argc > 3 && std::string(argv[3]) == "-Pareto");

    try {
        print("Loading safetensors via memory map...\n");
        MappedFile mapped_file(argv[1]);
        auto tensors = load_safe_tensors(mapped_file);

        print("Building dictionary for learned compression...\n");
        build_dictionary(tensors);

        std::sort(tensors.begin(), tensors.end(), [](const Tensor& a, const Tensor& b) {
            return a.name < b.name;
        });

        print("Compressing %zu tensors using 100%% lossless methods...\n", tensors.size());
        for (auto& t : tensors) {
            compress_tensor(t, use_pareto);
        }
        
        print("\nWriting Lossless GGUF...\n");
        write_gguf(argv[2], tensors);
        
        size_t orig_total = 0;
        size_t comp_total = 0;
        for(const auto& t : tensors) {
            orig_total += t.orig_bytes;
            comp_total += t.comp_bytes;
        }

        print("Done. Overall lossless compression ratio %.2f:1 (%.2f GB -> %.2f GB)\n",
             (comp_total > 0) ? double(orig_total) / comp_total : 0, orig_total / 1e9, comp_total / 1e9);
    } catch (const std::exception& e) {
        fprintf(stderr, "\nFATAL: %s\n", e.what());
        return 1;
    }
    return 0;
}
'@

$code | Out-File -FilePath $converterCpp -Encoding utf8

$jsonSingle = Join-Path $toolsDir 'json.hpp'
if (-not (Test-Path $jsonSingle)) {
    Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp' -OutFile $jsonSingle
}

Log 'Building lossless_converter.exe…'
Push-Location $toolsDir
$inc = @(
    "-I$(Join-Path $llamaSrc 'ggml\include')",
    "-I$(Join-Path $llamaSrc 'ggml\src')",
    "-I$srcDir\lz4\lib",
    "-I$srcDir\brotli\c\include",
    "-I$toolsDir"
)
$libs = @(
    "$buildDir\ggml.lib",
    "$buildDir\llama.lib",
    "$lz4Lib",
    "$brotliEncLib",
    (Join-Path $buildDir 'brotlicommon.lib'),
    'kernel32.lib','user32.lib','advapi32.lib'
)
$batCmd = @"
call "$vcvars" >nul
cl /std:c++17 /O2 /MD /EHsc /arch:AVX2 $inc "$converterCpp" /Fe:"$converterExe" /link $libs
"@
cmd /c $batCmd
if ($LASTEXITCODE) { throw 'Build failed' }
Pop-Location

# --- GUI and Job Management ---
$tbModel = New-Object System.Windows.Forms.TextBox
$tbModel.Dock = 'Top'
$tbModel.Height = 25
$form.Controls.Add($tbModel)

$btnBrowse = New-Object System.Windows.Forms.Button
$btnBrowse.Text = 'Browse…'
$btnBrowse.Dock = 'Top'
$btnBrowse.Height = 30
$btnBrowse.Add_Click({
    $d = New-Object System.Windows.Forms.OpenFileDialog
    $d.Filter = 'SafeTensors|*.safetensors|All|*.*'
    if ($d.ShowDialog() -eq 'OK') { $tbModel.Text = $d.FileName }
})
$form.Controls.Add($btnBrowse)

$flow = New-Object System.Windows.Forms.FlowLayoutPanel
$flow.Dock = 'Top'
$flow.Height = 40
$form.Controls.Add($flow)

$chkPareto = New-Object System.Windows.Forms.CheckBox
$chkPareto.Text = 'Pareto Opt (Find best LZ4 level, slower)'
$chkPareto.Checked = $Pareto
$chkPareto.AutoSize = $true
$flow.Controls.Add($chkPareto)

$btnConvert = New-Object System.Windows.Forms.Button
$btnConvert.Text = 'Convert → Lossless GGUF'
$btnConvert.Height = 30
$btnConvert.AutoSize = $true
$flow.Controls.Add($btnConvert)

$btnCancel = New-Object System.Windows.Forms.Button
$btnCancel.Text = 'Cancel'
$btnCancel.Height = 30
$btnCancel.Enabled = $false
$flow.Controls.Add($btnCancel)

$prg = New-Object System.Windows.Forms.ProgressBar
$prg.Dock = 'Top'
$prg.Style = 'Continuous'
$form.Controls.Add($prg)

$lblVram = New-Object System.Windows.Forms.Label
$lblVram.Dock = 'Top'
$lblVram.Height = 25
$form.Controls.Add($lblVram)

$global:job = $null
$btnConvert.Add_Click({
    $model = $tbModel.Text.Trim()
    if (-not (Test-Path $model)) { Log 'Model file not found'; return }
    $outName = [IO.Path]::GetFileNameWithoutExtension($model) + '_LOSSLESS.gguf'
    $outFile = Join-Path $outDir $outName
    $prg.Value = 0
    $btnConvert.Enabled = $false
    $btnCancel.Enabled = $true
    Log "Starting lossless conversion → $outFile"
    $paretoArg = if ($chkPareto.Checked) { '-Pareto' } else { '' }
    $global:job = Start-Job -FilePath $converterExe -ArgumentList $model, $outFile, $paretoArg
    $timer.Start()
})

$btnCancel.Add_Click({
    if ($global:job) {
        Log 'Cancelling…'
        $global:job | Stop-Job -Force
        $global:job | Remove-Job -Force
        $global:job = $null
    }
    $timer.Stop()
    $prg.Value = 0
    $btnConvert.Enabled = $true
    $btnCancel.Enabled = $false
})

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 200
$timer.Add_Tick({
    if (-not $global:job) { return }
    if ($global:job.State -eq 'Running') {
        $log = $global:job | Receive-Job
        if ($log) {
            foreach ($l in $log) { 
                Log $l
                if ($l -match '\[(\d+)/(\d+)\]') {
                    $done = [int]$Matches[1]
                    $total = [int]$Matches[2]
                    if ($total -gt 0) { $prg.Value = [int](100 * $done / $total) }
                }
            }
        }
    } else {
        $timer.Stop()
        $log = $global:job | Receive-Job
        if ($log) { foreach ($l in $log) { Log $l } }
        if ($global:job.State -eq 'Failed') { Log 'Conversion failed – see log above' }
        else {
            $outFile = Join-Path $outDir ([IO.Path]::GetFileNameWithoutExtension($tbModel.Text) + '_LOSSLESS.gguf')
            if (Test-Path $outFile) {
                Log "Success – output: $outFile"
                Log "NOTE: This is a specialized GGUF. A custom runtime is required for decompression."
            }
        }
        $global:job | Remove-Job -Force
        $global:job = $null
        $btnConvert.Enabled = $true
        $btnCancel.Enabled = $false
        $prg.Value = 100
    }
})

Log 'Lossless GGUF Converter ready – select model and click Convert'
$form.Add_FormClosed({ if ($global:job) { $global:job | Stop-Job -Force; $global:job | Remove-Job -Force } })
[void]$form.ShowDialog()
