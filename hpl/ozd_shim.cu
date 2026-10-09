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
//   HPL_OZD_PIPE    1 = split C into column chunks and overlap host->device
//                   copy, GEMM and device->host copy on three streams (default 1)
//   HPL_OZD_CHUNK   chunk width in columns for HPL_OZD_PIPE (default 2048)
//   HPL_OZD_SAFETY  0 = skip ozdgemm's non-finite input check (diagnostic;
//                   default 1)
// At exit, rank-local statistics are printed to stderr.
#include "ozdgemm.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" void dgemm_(const char*, const char*, const int*, const int*, const int*,
                       const double*, const double*, const int*, const double*,
                       const int*, const double*, double*, const int*);

namespace {

enum Mode { CPU, NATIVE, EMU, AUTO };

struct Stats {
    long   cpu_calls = 0, gpu_calls = 0, emu_calls = 0;
    double cpu_flop = 0, gpu_flop = 0;
    double t_cpu = 0, t_h2d = 0, t_gemm = 0, t_d2h = 0;  // seconds, per-stream busy time
    double t_wall = 0;  // wall time of the offloaded calls (copies + GEMM, overlapped or not)
    double bytes = 0;
    long   late_allocs = 0;  // device buffer (re)allocations after HPL_ozd_register
    int    moduli_min = 99, moduli_max = 0;
    long   fallbacks[5] = {0, 0, 0, 0, 0};
};

struct Shim {
    Mode          mode = EMU;
    int64_t       min_dim = 1024;
    bool          reg = true;
    bool          pipe = true;
    int           chunk = 2048;
    ozd_handle_t  h = nullptr;
    cudaStream_t  st = nullptr;           // compute (ozdgemm's stream)
    cudaStream_t  sin = nullptr, sout = nullptr;  // copies in / out (pipelined path)
    cudaEvent_t   ev[4];
    std::vector<cudaEvent_t> pev;         // 6 timing/sync events per chunk
    double*       dA = nullptr;
    double*       dB = nullptr;
    double*       dC = nullptr;
    size_t        cap[3] = {0, 0, 0};
    Stats         s;
    bool          init = false;
    bool          prealloc = false;  // HPL_ozd_register sized the buffers up front

    void setup() {
        init = true;
        const char* m = std::getenv("HPL_OZD_MODE");
        std::string ms = m ? m : "emu";
        mode = ms == "cpu" ? CPU : ms == "native" ? NATIVE : ms == "auto" ? AUTO : EMU;
        if (const char* v = std::getenv("HPL_OZD_MIN")) min_dim = std::atoll(v);
        if (const char* v = std::getenv("HPL_OZD_REG")) reg = std::atoi(v) != 0;
        if (const char* v = std::getenv("HPL_OZD_PIPE")) pipe = std::atoi(v) != 0;
        if (const char* v = std::getenv("HPL_OZD_CHUNK")) chunk = std::max(128, std::atoi(v));
        if (mode == CPU) return;
        cudaStreamCreate(&st);
        cudaStreamCreate(&sin);
        cudaStreamCreate(&sout);
        for (auto& e : ev) cudaEventCreate(&e);
        ozd_create(&h, st);
        ozd_set_max_workspace(h, size_t(24) << 30);
        ozd_set_backend(h, mode == NATIVE ? OZD_BACKEND_NATIVE
                           : mode == AUTO ? OZD_BACKEND_AUTO : OZD_BACKEND_EMU);
        if (const char* v = std::getenv("HPL_OZD_SAFETY")) ozd_set_safety_check(h, std::atoi(v) != 0);
        if (const char* v = std::getenv("HPL_OZD_TOL")) ozd_set_accuracy(h, std::atof(v));
        else ozd_set_moduli(h, std::getenv("HPL_OZD_MODULI") ? std::atoi(std::getenv("HPL_OZD_MODULI")) : 14);
        std::atexit([] { report(); });
    }

    static void report();

    void events(size_t n) {
        if (prealloc && pev.size() < n) s.late_allocs++;
        while (pev.size() < n) {
            cudaEvent_t e;
            cudaEventCreate(&e);
            pev.push_back(e);
        }
    }

    void grow(int i, size_t bytes) {
        if (bytes <= cap[i]) return;
        if (prealloc) s.late_allocs++;
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
                 "OZD_HPL mode=%s pipe=%d chunk=%d calls cpu=%ld gpu=%ld (emulated %ld, moduli %d..%d) "
                 "fallbacks[nonfinite,degenerate,dispatch,accuracy]=%ld,%ld,%ld,%ld "
                 "late_allocs=%ld\n"
                 "OZD_HPL gflop cpu=%.1f gpu=%.1f | time s: cpu_blas=%.2f h2d=%.2f gemm=%.2f "
                 "d2h=%.2f wall=%.2f | pcie GB=%.1f\n",
                 mode, g.pipe, g.chunk, s.cpu_calls, s.gpu_calls, s.emu_calls,
                 s.emu_calls ? s.moduli_min : 0, s.moduli_max, s.fallbacks[1], s.fallbacks[2],
                 s.fallbacks[3], s.fallbacks[4], s.late_allocs, s.cpu_flop / 1e9, s.gpu_flop / 1e9, s.t_cpu,
                 s.t_h2d, s.t_gemm, s.t_d2h, s.t_wall, s.bytes / 1e9);
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

// Library statistics of one HPL call, merged over its chunks: the call counts
// as emulated (or as a fallback of some reason) if any chunk was.
struct CallInfo {
    bool emu = false;
    bool fb[5] = {false, false, false, false, false};

    // After each ozd_dgemm: its info is filled in on the host before it returns.
    void add() {
        ozd_info_t info;
        ozd_get_info(g.h, &info);
        if (info.path == 1) {
            emu = true;
            g.s.moduli_min = std::min(g.s.moduli_min, info.moduli);
            g.s.moduli_max = std::max(g.s.moduli_max, info.moduli);
        }
        if (info.fallback_reason > 0 && info.fallback_reason < 5) fb[info.fallback_reason] = true;
    }
    void commit() const {
        g.s.emu_calls += emu;
        for (int r = 1; r < 5; ++r) g.s.fallbacks[r] += fb[r];
    }
};

// Copy everything, one GEMM, copy C back, all on one stream.
void gemm_serial(int ta, int tb, int M, int N, int K, double alpha, const double* A, int lda,
                 const double* B, int ldb, double beta, double* C, int ldc, int ar, int ac,
                 int br, int bc) {
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
    CallInfo ci;
    ci.add();
    ci.commit();
}

// C in column chunks of width w. A and B are copied first on the copy-in
// stream; then chunk j's C goes up on sin, its GEMM runs on st (ozdgemm's
// stream, so one workspace), and it comes back on sout. Chunks use disjoint
// slices of dB/dC, so the only dependencies are in -> gemm -> out per chunk.
// Output column j depends only on column j of op(B) and on A's row scaling,
// so the result equals the unchunked call (fixed moduli or native).
void gemm_pipelined(int ta, int tb, int M, int N, int K, double alpha, const double* A, int lda,
                    const double* B, int ldb, double beta, double* C, int ldc, int ar, int ac,
                    int br, int bc) {
    const int w = g.chunk, nch = (N + w - 1) / w;
    g.events(6 * (size_t)nch + 2);
    cudaEvent_t* e = g.pev.data();  // [0],[1]: A/B copy; chunk j: 2+6j .. 7+6j
    ozd_reserve(g.h, M, std::min(N, w), K);

    cudaEventRecord(e[0], g.sin);
    copy2d(g.dA, A, ar, ac, lda, cudaMemcpyHostToDevice, g.sin, false);
    copy2d(g.dB, B, br, bc, ldb, cudaMemcpyHostToDevice, g.sin, false);
    cudaEventRecord(e[1], g.sin);
    CallInfo ci;
    for (int j = 0; j < nch; ++j) {
        const int c0 = j * w, cw = std::min(w, N - c0);
        cudaEvent_t* ej = e + 2 + 6 * j;  // in0 in1 gemm0 gemm1 out0 out1
        double* dCj = g.dC + (size_t)c0 * M;
        cudaEventRecord(ej[0], g.sin);
        if (beta != 0.0)
            copy2d(dCj, C + (size_t)c0 * ldc, M, cw, ldc, cudaMemcpyHostToDevice, g.sin, false);
        cudaEventRecord(ej[1], g.sin);

        cudaStreamWaitEvent(g.st, ej[1], 0);
        cudaEventRecord(ej[2], g.st);
        const double* dBj = tb == 111 ? g.dB + (size_t)c0 * br : g.dB + c0;
        ozd_dgemm(g.h, ta == 111 ? OZD_OP_N : OZD_OP_T, tb == 111 ? OZD_OP_N : OZD_OP_T, M, cw, K,
                  alpha, g.dA, ar, dBj, br, beta, dCj, M);
        cudaEventRecord(ej[3], g.st);
        ci.add();

        cudaStreamWaitEvent(g.sout, ej[3], 0);
        cudaEventRecord(ej[4], g.sout);
        copy2d(C + (size_t)c0 * ldc, dCj, M, cw, ldc, cudaMemcpyDeviceToHost, g.sout, true);
        cudaEventRecord(ej[5], g.sout);
    }
    cudaStreamSynchronize(g.sout);
    ci.commit();

    float t;
    cudaEventElapsedTime(&t, e[0], e[1]);
    g.s.t_h2d += t * 1e-3;
    for (int j = 0; j < nch; ++j) {
        cudaEvent_t* ej = e + 2 + 6 * j;
        cudaEventElapsedTime(&t, ej[0], ej[1]); g.s.t_h2d += t * 1e-3;
        cudaEventElapsedTime(&t, ej[2], ej[3]); g.s.t_gemm += t * 1e-3;
        cudaEventElapsedTime(&t, ej[4], ej[5]); g.s.t_d2h += t * 1e-3;
    }
}

}  // namespace

extern "C" {

// Called by the patched HPL_pdtest after the matrix allocation, before the
// timed solve. Page-locks the matrix and allocates every device buffer, the
// ozdgemm workspace and the chunk events for the largest update this rank can
// see (local mp x nq trailing matrix, K = nb); the trailing matrix only
// shrinks, so no offloaded call allocates (late_allocs in the report).
void HPL_ozd_register(void* p, size_t bytes, int mp, int nq, int nb) {
    if (!g.init) g.setup();
    if (g.mode == CPU) return;
    if (g.reg && p) cudaHostRegister(p, bytes, cudaHostRegisterDefault);
    if (mp <= 0 || nq <= 0 || nb <= 0) return;
    g.grow(0, (size_t)mp * nb * 8);
    g.grow(1, (size_t)nb * nq * 8);
    g.grow(2, (size_t)mp * nq * 8);
    if (g.pipe) {
        ozd_reserve(g.h, mp, std::min(nq, g.chunk), nb);
        g.events(6 * (size_t)((nq + g.chunk - 1) / g.chunk) + 2);
    } else {
        ozd_reserve(g.h, mp, nq, nb);
    }
    cudaDeviceSynchronize();
    g.prealloc = true;
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
    const double t0 = now();
    if (g.pipe && N > g.chunk) {
        gemm_pipelined(ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, ar, ac, br, bc);
    } else {
        gemm_serial(ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, ar, ac, br, bc);
    }
    g.s.t_wall += now() - t0;
    g.s.bytes += 8.0 * ((double)ar * ac + (double)br * bc + (double)M * N * (beta != 0.0 ? 2 : 1));
    g.s.gpu_calls++;
    g.s.gpu_flop += flop;
}

}  // extern "C"
