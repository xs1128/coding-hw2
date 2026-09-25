// End-to-end GEMM benchmark: native cuBLAS DGEMM vs ozdgemm emulation.
//
// Timing is end-to-end per call (scaling, safety check, conversion, INT8 GEMMs,
// reduction, CRT), measured with CUDA events. The workspace is reserved before
// timing so nothing is allocated in the timed region. Each shape reports the
// median and min over --reps calls after --warmup calls.
//
// Usage: bench [--preset square|hpl|skinny|all] [--shape m,n,k]...
//              [--moduli 14[,11,...]] [--reps 10] [--warmup 3] [--no-native]
//              [--breakdown] [--nosafety]
// Output: CSV on stdout.
#include "../tests/common.cuh"
#include "ozdgemm.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <unistd.h>

struct Shape { int64_t m, n, k; };

static std::vector<Shape> preset(const std::string& p) {
    std::vector<Shape> s;
    if (p == "square" || p == "all")
        for (int64_t x : {1024, 2048, 4096, 8192, 12288, 16384}) s.push_back({x, x, x});
    if (p == "hpl" || p == "all")
        for (int64_t x : {8192, 16384, 32768})
            for (int64_t k : {256, 512, 1024, 2048}) s.push_back({x, x, k});
    if (p == "skinny" || p == "all") {
        s.push_back({16384, 64, 16384});
        s.push_back({16384, 256, 16384});
        s.push_back({65536, 1024, 1024});
    }
    return s;
}

int main(int argc, char** argv) {
    std::vector<Shape> shapes;
    std::vector<int> moduli = {14};
    int reps = 10, warmup = 3;
    bool native = true, breakdown = false, safety = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--preset") { auto p = preset(next()); shapes.insert(shapes.end(), p.begin(), p.end()); }
        else if (a == "--shape") {
            Shape s; char c;
            std::istringstream(next()) >> s.m >> c >> s.n >> c >> s.k;
            shapes.push_back(s);
        } else if (a == "--moduli") {
            moduli.clear();
            std::istringstream ss(next());
            for (std::string t; std::getline(ss, t, ',');) moduli.push_back(std::atoi(t.c_str()));
        }
        else if (a == "--reps") reps = std::atoi(next().c_str());
        else if (a == "--warmup") warmup = std::atoi(next().c_str());
        else if (a == "--no-native") native = false;
        else if (a == "--breakdown") breakdown = true;
        else if (a == "--nosafety") safety = false;
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (shapes.empty()) shapes = preset("square");

    char host[64] = {0};
    gethostname(host, sizeof host - 1);
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    std::string gpu = prop.name;
    std::replace(gpu.begin(), gpu.end(), ' ', '_');

    ozd_handle_t h;
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    if (ozd_create(&h, st) != OZD_OK) return 2;
    ozd_set_safety_check(h, safety);
    ozd_set_max_workspace(h, size_t(16) << 30);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    std::printf("host,gpu,backend,moduli,m,n,k,median_ms,min_ms,tflops_median,col_blocks,"
                "scale_ms,convert_ms,gemm_ms,reduce_ms,crt_ms\n");
    for (const auto& s : shapes) {
        std::vector<double> A, B;
        fill(A, s.m, s.k, s.m, Gen::parse("uniform"), 1, true);
        fill(B, s.k, s.n, s.k, Gen::parse("uniform"), 2, false);
        double *dA, *dB, *dC;
        CK(cudaMalloc(&dA, A.size() * 8));
        CK(cudaMalloc(&dB, B.size() * 8));
        CK(cudaMalloc(&dC, (size_t)s.m * s.n * 8));
        CK(cudaMemcpy(dA, A.data(), A.size() * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, B.data(), B.size() * 8, cudaMemcpyHostToDevice));
        CK(cudaMemset(dC, 0, (size_t)s.m * s.n * 8));

        std::vector<std::pair<int, int>> runs;  // (backend, moduli)
        if (native) runs.push_back({OZD_BACKEND_NATIVE, 0});
        for (int md : moduli) runs.push_back({OZD_BACKEND_EMU, md});
        for (auto [bk, md] : runs) {
            ozd_set_backend(h, (ozd_backend_t)bk);
            if (md) ozd_set_moduli(h, md);
            if (bk == OZD_BACKEND_EMU && ozd_reserve(h, s.m, s.n, s.k) != OZD_OK) {
                std::fprintf(stderr, "reserve failed for %ld,%ld,%ld\n", (long)s.m, (long)s.n, (long)s.k);
                continue;
            }
            auto call = [&]() {
                return ozd_dgemm(h, OZD_OP_N, OZD_OP_N, s.m, s.n, s.k, 1.0, dA, s.m, dB, s.k,
                                 0.0, dC, s.m);
            };
            for (int w = 0; w < warmup; ++w) call();
            CK(cudaStreamSynchronize(st));
            std::vector<double> t;
            for (int r = 0; r < reps; ++r) {
                CK(cudaEventRecord(e0, st));
                if (call() != OZD_OK) { std::fprintf(stderr, "ozd_dgemm failed\n"); return 2; }
                CK(cudaEventRecord(e1, st));
                CK(cudaEventSynchronize(e1));
                float ms;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                t.push_back(ms);
            }
            std::sort(t.begin(), t.end());
            const double med = t[t.size() / 2], mn = t[0];
            ozd_info_t info{};
            if (breakdown && bk == OZD_BACKEND_EMU) {
                // Separate, untimed call with per-stage events.
                ozd_set_profiling(h, 1);
                call();
                ozd_set_profiling(h, 0);
            }
            ozd_get_info(h, &info);
            const double tf = 2.0 * s.m * s.n * s.k / (med * 1e-3) / 1e12;
            std::printf("%s,%s,%s,%d,%ld,%ld,%ld,%.4f,%.4f,%.2f,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n", host,
                        gpu.c_str(), bk == OZD_BACKEND_NATIVE ? "native" : "emu_v0", md, (long)s.m,
                        (long)s.n, (long)s.k, med, mn, tf, info.col_blocks, info.t_scale_ms,
                        info.t_convert_ms, info.t_gemm_ms, info.t_reduce_ms, info.t_crt_ms);
            std::fflush(stdout);
        }
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    ozd_destroy(h);
    return 0;
}
