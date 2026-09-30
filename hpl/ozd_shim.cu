// Link-time DGEMM hook for netlib HPL 2.3.
//
// HPL built with -DHPL_CALL_CBLAS maps HPL_dgemm to cblas_dgemm. This file
// defines cblas_dgemm itself, so it overrides the CPU BLAS: small calls go to
// the CPU BLAS (Fortran dgemm_), large ones are copied to the GPU and run
// through ozdgemm. HPL's matrix stays in host memory (netlib HPL is a CPU
// code), so every offloaded call moves A, B and C over PCIe.
//
// Environment:
//   HPL_OZD_MODE    cpu | native | emu | auto       (default emu)
//                   native = cuBLAS DGEMM through the same offload path
//   HPL_OZD_MODULI  fixed moduli count (default 14)
//   HPL_OZD_TOL     adaptive tolerance (overrides MODULI; 0 = DGEMM-level)
//   HPL_OZD_MIN     offload when M, N >= MIN and K >= MIN/8 (default 1024)
//   HPL_OZD_REG     1 = page-lock HPL's matrix via HPL_ozd_register (default 1)
// At exit, rank-local statistics are printed to stderr.
#include "ozdgemm.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" void dgemm_(const char*, const char*, const int*, const int*, const int*,
                       const double*, const double*, const int*, const double*,
                       const int*, const double*, double*, const int*);

namespace {

enum Mode { CPU, NATIVE, EMU, AUTO };

struct Stats {
    long   cpu_calls = 0, gpu_calls = 0, emu_calls = 0;
    double cpu_flop = 0, gpu_flop = 0;
    double t_cpu = 0, t_h2d = 0, t_gemm = 0, t_d2h = 0;  // seconds
    double bytes = 0;
    int    moduli_min = 99, moduli_max = 0;
    long   fallbacks[5] = {0, 0, 0, 0, 0};
};

struct Shim {
    Mode          mode = EMU;
    int64_t       min_dim = 1024;
    bool          reg = true;
    ozd_handle_t  h = nullptr;
    cudaStream_t  st = nullptr;
    cudaEvent_t   ev[4];
    double*       dA = nullptr;
    double*       dB = nullptr;
    double*       dC = nullptr;
    size_t        cap[3] = {0, 0, 0};
    Stats         s;
    bool          init = false;

    void setup() {
        init = true;
        const char* m = std::getenv("HPL_OZD_MODE");
        std::string ms = m ? m : "emu";
        mode = ms == "cpu" ? CPU : ms == "native" ? NATIVE : ms == "auto" ? AUTO : EMU;
        if (const char* v = std::getenv("HPL_OZD_MIN")) min_dim = std::atoll(v);
        if (const char* v = std::getenv("HPL_OZD_REG")) reg = std::atoi(v) != 0;
        if (mode == CPU) return;
        cudaStreamCreate(&st);
        for (auto& e : ev) cudaEventCreate(&e);
        ozd_create(&h, st);
        ozd_set_max_workspace(h, size_t(24) << 30);
        ozd_set_backend(h, mode == NATIVE ? OZD_BACKEND_NATIVE
                           : mode == AUTO ? OZD_BACKEND_AUTO : OZD_BACKEND_EMU);
        if (const char* v = std::getenv("HPL_OZD_TOL")) ozd_set_accuracy(h, std::atof(v));
        else ozd_set_moduli(h, std::getenv("HPL_OZD_MODULI") ? std::atoi(std::getenv("HPL_OZD_MODULI")) : 14);
        std::atexit([] { report(); });
    }

    static void report();

    void grow(int i, size_t bytes) {
        if (bytes <= cap[i]) return;
        double** p = i == 0 ? &dA : i == 1 ? &dB : &dC;
        if (*p) cudaFree(*p);
        cudaMalloc(p, bytes);
        cap[i] = bytes;
    }
};

Shim g;

void Shim::report() {
    const Stats& s = g.s;
    const char* mode = g.mode == NATIVE ? "native" : g.mode == AUTO ? "auto" : "emu";
    std::fprintf(stderr,
                 "OZD_HPL mode=%s calls cpu=%ld gpu=%ld (emulated %ld, moduli %d..%d) "
                 "fallbacks[nonfinite,degenerate,dispatch,accuracy]=%ld,%ld,%ld,%ld\n"
                 "OZD_HPL gflop cpu=%.1f gpu=%.1f | time s: cpu_blas=%.2f h2d=%.2f gemm=%.2f "
                 "d2h=%.2f | pcie GB=%.1f (%.1f GB/s)\n",
                 mode, s.cpu_calls, s.gpu_calls, s.emu_calls,
                 s.emu_calls ? s.moduli_min : 0, s.moduli_max, s.fallbacks[1], s.fallbacks[2],
                 s.fallbacks[3], s.fallbacks[4], s.cpu_flop / 1e9, s.gpu_flop / 1e9, s.t_cpu,
                 s.t_h2d, s.t_gemm, s.t_d2h, s.bytes / 1e9,
                 s.bytes / 1e9 / (s.t_h2d + s.t_d2h + 1e-12));
}

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Copy an r x c column-major block with leading dimension ld (elements).
void copy2d(double* dst, const double* src, int r, int c, int ld, cudaMemcpyKind kind,
            cudaStream_t st, bool to_host) {
    if (to_host)
        cudaMemcpy2DAsync(dst, (size_t)ld * 8, src, (size_t)r * 8, (size_t)r * 8, c, kind, st);
    else
        cudaMemcpy2DAsync(dst, (size_t)r * 8, src, (size_t)ld * 8, (size_t)r * 8, c, kind, st);
}

}  // namespace

extern "C" {

// Called by the patched HPL_pdtest around the matrix allocation.
void HPL_ozd_register(void* p, size_t bytes) {
    if (!g.init) g.setup();
    if (g.mode != CPU && g.reg && p) cudaHostRegister(p, bytes, cudaHostRegisterDefault);
}
void HPL_ozd_unregister(void* p) {
    if (g.mode != CPU && g.reg && p) cudaHostUnregister(p);
}

// CBLAS enum values used by HPL: 101 row-major, 102 column-major,
// 111 no-transpose, 112 transpose.
void cblas_dgemm(int order, int ta, int tb, int M, int N, int K, double alpha,
                 const double* A, int lda, const double* B, int ldb, double beta,
                 double* C, int ldc) {
    if (!g.init) g.setup();
    if (order == 101) {  // C^T = op(B)^T op(A)^T
        cblas_dgemm(102, tb, ta, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc);
        return;
    }
    const double flop = 2.0 * M * N * K;
    if (g.mode == CPU || M < g.min_dim || N < g.min_dim || K < g.min_dim / 8) {
        const double t0 = now();
        const char ca = ta == 111 ? 'N' : 'T', cb = tb == 111 ? 'N' : 'T';
        dgemm_(&ca, &cb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        g.s.t_cpu += now() - t0;
        g.s.cpu_calls++;
        g.s.cpu_flop += flop;
        return;
    }
    const int ar = ta == 111 ? M : K, ac = ta == 111 ? K : M;
    const int br = tb == 111 ? K : N, bc = tb == 111 ? N : K;
    g.grow(0, (size_t)ar * ac * 8);
    g.grow(1, (size_t)br * bc * 8);
    g.grow(2, (size_t)M * N * 8);
    ozd_reserve(g.h, M, N, K);

    cudaEventRecord(g.ev[0], g.st);
    copy2d(g.dA, A, ar, ac, lda, cudaMemcpyHostToDevice, g.st, false);
    copy2d(g.dB, B, br, bc, ldb, cudaMemcpyHostToDevice, g.st, false);
    if (beta != 0.0) copy2d(g.dC, C, M, N, ldc, cudaMemcpyHostToDevice, g.st, false);
    cudaEventRecord(g.ev[1], g.st);
    ozd_dgemm(g.h, ta == 111 ? OZD_OP_N : OZD_OP_T, tb == 111 ? OZD_OP_N : OZD_OP_T, M, N, K,
              alpha, g.dA, ar, g.dB, br, beta, g.dC, M);
    cudaEventRecord(g.ev[2], g.st);
    copy2d(C, g.dC, M, N, ldc, cudaMemcpyDeviceToHost, g.st, true);
    cudaEventRecord(g.ev[3], g.st);
    cudaEventSynchronize(g.ev[3]);

    float t[3];
    for (int i = 0; i < 3; ++i) cudaEventElapsedTime(&t[i], g.ev[i], g.ev[i + 1]);
    g.s.t_h2d += t[0] * 1e-3;
    g.s.t_gemm += t[1] * 1e-3;
    g.s.t_d2h += t[2] * 1e-3;
    g.s.bytes += 8.0 * ((double)ar * ac + (double)br * bc + (double)M * N * (beta != 0.0 ? 2 : 1));
    g.s.gpu_calls++;
    g.s.gpu_flop += flop;
    ozd_info_t info;
    ozd_get_info(g.h, &info);
    if (info.path == 1) {
        g.s.emu_calls++;
        g.s.moduli_min = std::min(g.s.moduli_min, info.moduli);
        g.s.moduli_max = std::max(g.s.moduli_max, info.moduli);
    }
    if (info.fallback_reason > 0 && info.fallback_reason < 5) g.s.fallbacks[info.fallback_reason]++;
}

}  // extern "C"
