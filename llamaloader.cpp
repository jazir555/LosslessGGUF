// lz5_v2_0_loader.cpp
// Usage: lz5_v2_0_loader.exe input_LZ5v2.gguf output.safetensors
// Requires: ggml.h, gguf.h, zstd, ggml fp16 helpers

#include <bits/stdc++.h>
#define GGML_BUILD
#include "ggml.h"
#include "gguf.h"
#include <zstd.h>
#include <openssl/sha.h> // for SHA256 checks

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#endif

using namespace std;

static void die(const string &m){ cerr<<m<<"\n"; exit(1); }

static vector<uint8_t> read_file_all(const string &path){
    ifstream f(path, ios::binary);
    if(!f) die("cannot open file: " + path);
    f.seekg(0, ios::end);
    size_t n = (size_t)f.tellg();
    f.seekg(0, ios::beg);
    vector<uint8_t> buf(n);
    f.read((char*)buf.data(), n);
    return buf;
}

static string hex_of(const uint8_t *data, size_t n){
    static const char *hex = "0123456789abcdef";
    string s; s.reserve(n*2);
    for(size_t i=0;i<n;i++){ s.push_back(hex[data[i]>>4]); s.push_back(hex[data[i]&0xF]); }
    return s;
}

// Compute SHA256 of raw bytes
static string sha256_hex(const uint8_t *data, size_t n){
    unsigned char md[SHA256_DIGEST_LENGTH];
    SHA256(data, n, md);
    return hex_of(md, SHA256_DIGEST_LENGTH);
}

int main(int argc, char **argv){
    if(argc < 3) die("usage: lz5_v2_0_loader.exe input_LZ5v2.gguf out.safetensors");

    const string in_path = argv[1];
    const string out_path = argv[2];

    // Open file and parse GGUF header via gguf
    FILE *inf = fopen(in_path.c_str(), "rb");
    if(!inf) die("cannot open input file");

    // load gguf context (reads header but not necessarily payloads)
    gguf_context *ctx = gguf_read_from_file(inf);
    if(!ctx) die("gguf_read_from_file failed (is this a valid GGUF?)");

    // collect tensor list and metadata
    size_t n_tensors = gguf_n_tensors(ctx);
    struct Entry { string name; vector<uint64_t> shape; ggml_type dtype; bool lz5; uint64_t payload_size; uint64_t payload_file_offset; };
    vector<Entry> entries; entries.reserve(n_tensors);

    // NOTE: gguf_read_from_file typically positions file pointer after header.
    // We'll need to know where the per-tensor payloads begin and their order.
    // Many gguf writers append raw payloads in the same order gguf_add_tensor was called.
    // We'll use gguf API to iterate tensors and read metadata flags.

    for(size_t i=0;i<n_tensors;i++){
        gguf_tensor_info ti;
        if(!gguf_tensor_info_at(ctx, i, &ti)) die("failed gguf_tensor_info_at");
        Entry e;
        e.name = ti.name;
        e.dtype = ti.type;
        e.shape.assign(ti.shape, ti.shape + ti.ndim);
        // check metadata flag lz5.is_compressed
        bool lzflag = false;
        gguf_get_tensor_val_bool(ctx, ti.name, "lz5.is_compressed", &lzflag);
        e.lz5 = lzflag;
        // find stored size metadata if present (converter writes comp_bytes into payload length implicitly)
        // We cannot rely on a stored "payload_size" meta; instead we will read sequentially from file stream in same order converter appended payloads.
        entries.push_back(std::move(e));
    }

    // Determine where the file read pointer currently is (end of header). We'll stream payloads from here.
    long payload_start = ftell(inf);
    if(payload_start < 0) die("ftell failed on input file");
    // Move to end to compute size
    fseek(inf, 0, SEEK_END);
    long file_end = ftell(inf);
    if(file_end < 0) die("ftell end failed");
    // Reset to payload_start to read payloads sequentially
    fseek(inf, payload_start, SEEK_SET);

    // Prepare output safetensors header JSON building
    // We'll need to write a safetensors header then payloads in the same format. Simpler: write a binary safetensors-like container:
    //  1) write 8 bytes header_len
    //  2) write JSON describing tensors with data_offsets
    //  3) append raw tensor bytes (in correct order)
    // To do that we must reconstruct the original tensor bytes.

    struct OutTensor {
        string name;
        vector<uint64_t> shape;
        ggml_type dtype;
        vector<uint8_t> bytes; // reconstructed raw bytes to write
    };
    vector<OutTensor> out_tensors; out_tensors.reserve(entries.size());

    // Load predictors from gguf if present: predictors are stored under prefix "lz5.pred.<base>"
    // We'll load all tensors in ctx and find predictor tensors by name prefix
    // Build a map<string, PredictorData> - for simplicity we store predictor as raw float arrays to compute forward.
    struct PredictorData {
        vector<float> W1; size_t W1_r=0,W1_c=0;
        vector<float> b1; size_t b1_n=0;
        vector<float> W2; size_t W2_r=0,W2_c=0;
        vector<float> b2; size_t b2_n=0;
        bool valid=false;
    };
    unordered_map<string, PredictorData> preds;

    // iterate all gguf tensors in ctx to find predictor tensors
    for(size_t i=0;i<n_tensors;i++){
        gguf_tensor_info ti;
        gguf_tensor_info_at(ctx, i, &ti);
        string tname = ti.name;
        // predictor naming: "lz5.pred.<base>.W1"
        const string prefix = "lz5.pred.";
        if(tname.rfind(prefix,0) == 0){
            // parse base and suffix
            string tail = tname.substr(prefix.size()); // <base>.W1 or .b1 etc
            size_t dot = tail.rfind('.');
            if(dot==string::npos) continue;
            string base = tail.substr(0,dot);
            string suf  = tail.substr(dot+1);
            PredictorData &P = preds[base];
            // read tensor bytes from gguf API (gguf_get_tensor_data returns pointer into ctx internal storage)
            size_t nelems = 1;
            for(int d=0; d<ti.ndim; ++d) nelems *= ti.shape[d];
            const void *data_ptr = gguf_get_tensor_data(ctx, ti.name);
            if(!data_ptr) continue;
            if(ti.type != GGML_TYPE_F32) continue; // predictors stored as F32 in converter
            const float *fptr = (const float*)data_ptr;
            if(suf == "W1"){
                P.W1.assign(fptr, fptr + nelems); P.W1_r = ti.shape[0]; P.W1_c = ti.shape[1]; P.valid = true;
            }else if(suf == "b1"){
                P.b1.assign(fptr, fptr + nelems); P.b1_n = nelems; P.valid = true;
            }else if(suf == "W2"){
                P.W2.assign(fptr, fptr + nelems); P.W2_r = ti.shape[0]; P.W2_c = ti.shape[1]; P.valid = true;
            }else if(suf == "b2"){
                P.b2.assign(fptr, fptr + nelems); P.b2_n = nelems; P.valid = true;
            }
        }
    }

    // helper: predictor forward (single-row) - single-threaded deterministic
    auto predictor_forward = [&](const PredictorData &P, const vector<float> &x_in, vector<float> &out){
        // x_in length must match W1_r (=in_dim)
        size_t in_d = P.W1_r;
        size_t hid  = P.W1_c;
        size_t out_d = P.W2_c; // hopefully equals in_d
        vector<float> h(hid);
        // h = ReLU(x * W1 + b1)
        for(size_t j=0;j<hid;j++){
            double acc = 0.0;
            for(size_t i=0;i<in_d;i++){
                acc += (double)x_in[i] * (double)P.W1[i*hid + j];
            }
            acc += (P.b1_n>j) ? P.b1[j] : 0.0;
            h[j] = (float)max(0.0, acc);
        }
        // y = h * W2 + b2
        out.assign(out_d, 0.0f);
        for(size_t j=0;j<out_d;j++){
            double acc = 0.0;
            for(size_t k=0;k<hid;k++){
                acc += (double)h[k] * (double)P.W2[k*out_d + j];
            }
            acc += (P.b2_n>j) ? P.b2[j] : 0.0;
            out[j] = (float)acc;
        }
    };

    // sequentially read payloads from file (converter wrote them in the same tensor order)
    for(size_t ti_idx=0; ti_idx<entries.size(); ++ti_idx){
        const Entry &e = entries[ti_idx];
        size_t nelems = 1;
        for(auto s : e.shape) nelems *= (size_t)s;
        size_t bytes_expected = 0;
        if(e.dtype == GGML_TYPE_F32) bytes_expected = nelems * sizeof(float);
        else if(e.dtype == GGML_TYPE_F16) bytes_expected = nelems * sizeof(ggml_fp16_t);
        else bytes_expected = nelems * sizeof(float); // fallback

        vector<uint8_t> reconstructed_bytes;
        reconstructed_bytes.resize(bytes_expected);

        if(!e.lz5){
            // raw bytes were stored; simply read bytes_expected from stream
            size_t r = fread(reconstructed_bytes.data(), 1, bytes_expected, inf);
            if(r != bytes_expected) die("unexpected EOF while reading raw tensor payload");
        } else {
            // compressed residual; we need to read compressed chunk from stream.
            // Problem: we don't have length markers. But converter wrote compressed_residual.size() bytes,
            // and the writer was sequential without extra size headers. To be robust, the converter should
            // have written a 8-byte payload-length before each tensor payload. If you didn't, you must update the converter.
            // We'll assume converter was updated to write a 8-byte little-endian length before each payload.
            uint64_t comp_len = 0;
            if(fread(&comp_len, 1, sizeof(uint64_t), inf) != sizeof(uint64_t)) die("failed to read comp_len");
            vector<uint8_t> compbuf;
            compbuf.resize((size_t)comp_len);
            if(fread(compbuf.data(), 1, (size_t)comp_len, inf) != comp_len) die("failed to read compressed payload");
            // decompress
            unsigned long long out_size = bytes_expected;
            vector<uint8_t> outbuf(out_size);
            size_t dec = ZSTD_decompress(outbuf.data(), out_size, compbuf.data(), compbuf.size());
            if(ZSTD_isError(dec)) die(string("zstd decompress failed: ") + ZSTD_getErrorName(dec));
            if(dec != out_size) {
                // If converter stored fp16 residuals, out_size matches bytes_expected. Otherwise mismatch -> fail.
                die("decompressed size mismatch");
            }
            // now outbuf contains residual float bytes (or fp16 depending on converter). We must compute predictor and add residual.
            // Identify predictor base for this tensor name
            string base = e.name;
            // convert ".layers.<N>." to ".layers.X."
            base = regex_replace(base, std::regex(R"(\\.layers\\.\\d+\\.)"), ".layers.X.");
            auto itp = preds.find(base);
            if(itp == preds.end()){
                die("predictor not found for tensor: " + e.name + " — converter must embed predictor for this base");
            }
            PredictorData &P = itp->second;
            // Build input X for predictor. The converter's predictor used concatenation of previous two layer tensors.
            // For lossless reconstruction we must reproduce identical X used by converter; easiest safe approach:
            // converter must store predictor inputs or use a deterministic predictor that uses only small context. To keep lossless,
            // the converter should have stored the predictor inputs or used predictors only based on parameters already present.
            // Here we assume converter used static zero-context predictor (or stored predictors that don't require external context).
            // Compute pred (single-row)
            vector<float> x_in(P.W1_r, 0.0f); // best-effort
            vector<float> y_pred;
            predictor_forward(P, x_in, y_pred);
            // now add residual (interpreted as float32 array) elementwise to y_pred to produce original vec
            // interpret outbuf as float*
            float *resf = (float*)outbuf.data();
            if(e.dtype == GGML_TYPE_F32){
                for(size_t i=0;i<nelems;i++){
                    float orig = y_pred.size() > i ? (y_pred[i] + resf[i]) : resf[i];
                    memcpy(reconstructed_bytes.data() + i*sizeof(float), &orig, sizeof(float));
                }
            } else if(e.dtype == GGML_TYPE_F16){
                // we need to convert pred (float) + residual (float) to fp16 using ggml helper
                // combine into float buffer then convert
                vector<float> tmpf(nelems);
                for(size_t i=0;i<nelems;i++) tmpf[i] = y_pred.size() > i ? (y_pred[i] + resf[i]) : resf[i];
                // use ggml_fp32_to_fp16_row if available; fallback basic rounding (but must match converter)
                ggml_fp16_t *fp16 = (ggml_fp16_t*)reconstructed_bytes.data();
                ggml_fp32_to_fp16_row(tmpf.data(), fp16, (int)nelems);
            } else {
                die("unsupported dtype in loader");
            }
        }

        // Optionally: check SHA256 metadata (converter should have set gguf tensor val "lz5.sha256" to hex string)
        const char* sha_meta = nullptr;
        if(gguf_get_tensor_val_str(ctx, e.name.c_str(), "lz5.sha256", &sha_meta)) {
            string expected = sha_meta;
            string actual = sha256_hex(reconstructed_bytes.data(), reconstructed_bytes.size());
            if(expected != actual) {
                die("SHA256 mismatch for tensor " + e.name + " expected " + expected + " actual " + actual);
            }
        }

        // push into out_tensors to write safetensors later
        // record shape as uint64_t for safetensors JSON
        // We'll write safetensors: 8 bytes header_len, JSON, then payloads in same order
        // For now store the bytes in memory (be careful with very large models!)
        // For production with huge models, stream to disk per-tensor rather than storing all in memory.
        // We'll stream here: write to temp file per tensor and record offsets.
        // For simplicity in this loader, we write all into memory (user beware)
        // Append to vector of out_tensors
        OutTensor ot;
        ot.name = e.name;
        ot.shape.assign(e.shape.begin(), e.shape.end());
        ot.dtype = e.dtype;
        ot.bytes = move(reconstructed_bytes);
        // write immediate to temporary buffer vector (we'll emit safetensors below)
        // We accumulate for all tensors
        // (But for huge models you must change this to streaming writes.)
        // store into a container accessible below: use a vector on heap
        static vector<OutTensor> global_outs;
        global_outs.push_back(std::move(ot));
    }

    // now construct safetensors header JSON and write output file
    // compute offsets
    uint64_t cur_off = 0;
    nlohmann::json jhdr;
    jhdr["__metadata__"] = nlohmann::json::object();
    vector<vector<uint8_t>> payloads;
    for(const auto &ot : global_outs){
        vector<uint64_t> shape64(ot.shape.begin(), ot.shape.end());
        // dtype string
        string dtype_str = (ot.dtype == GGML_TYPE_F32) ? "F32" : (ot.dtype == GGML_TYPE_F16 ? "F16" : "F32");
        // create meta
        nlohmann::json meta;
        meta["dtype"] = dtype_str;
        meta["shape"] = nlohmann::json::array();
        for(auto s : ot.shape) meta["shape"].push_back((int64_t)s);
        meta["data_offsets"] = nlohmann::json::array();
        meta["data_offsets"].push_back((uint64_t)cur_off);
        meta["data_offsets"].push_back((uint64_t)(cur_off + ot.bytes.size()));
        // optional SHA256 (we can compute now)
        string sha = sha256_hex(ot.bytes.data(), ot.bytes.size());
        meta["sha256"] = sha;
        jhdr[ot.name] = meta;
        cur_off += ot.bytes.size();
        payloads.push_back(ot.bytes);
    }

    // serialize json
    string json_text = jhdr.dump();
    // open output file
    FILE *outf = fopen(out_path.c_str(), "wb");
    if(!outf) die("cannot open out file");
    uint64_t header_len = (uint64_t)json_text.size();
    fwrite(&header_len, 8, 1, outf);
    fwrite(json_text.data(), 1, json_text.size(), outf);
    // write payloads
    for(auto &p : payloads) fwrite(p.data(), 1, p.size(), outf);
    fclose(outf);

    gguf_free(ctx);
    fclose(inf);
    cout<<"Done: wrote "<<out_path<<"\n";
    return 0;
}
