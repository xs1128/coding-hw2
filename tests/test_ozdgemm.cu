// Correctness tests for ozdgemm.
//
// Pass criterion (derived in README "Correctness standard"): for every entry,
//   |C - C_ref| <= 4 * sqrt(k) * 2^-P * |alpha| * ||A_i|| * ||B_j||
//                  + 4u * (|alpha| |A||B| + |beta C0|)_ij
// where P = ozd_scale_bits(moduli) and C_ref is a double-double GEMM.
// This checks the implementation against the method's own guarantee. We also
// print the componentwise error of native DGEMM for comparison.
//
// Adaptive cases (ozd_set_accuracy) must additionally meet the requested
// guarantee |C - C_ref| <= tol * |A||B| + 4u * (...), or report that the
// tolerance is unreachable and fall back to native DGEMM.
//
// Usage: test_ozdgemm              run the suite, exit 1 on any failure
//        test_ozdgemm --sweep      CSV of error vs. moduli (informational)
//        test_ozdgemm --adaptive   CSV of moduli chosen vs. tolerance
#include "common.cuh"
#include "ozdgemm.h"

#include <algorithm>
#include <cstring>

namespace {

struct Case {
    const char* name;
    int64_t m, n, k;
    int ta, tb;
    const char* gen;
    double alpha, beta;
    int moduli;
    int64_t pad = 0;           // extra leading-dimension padding
    size_t max_ws = 0;         // 0 = default
    double tol = -1;           // >= 0: adaptive moduli with this tolerance (0 = DGEMM-level)
    int expect_level = -1;     // >= 0: must be emulated after this |A||B| bound level
};

struct Result {
    double err_nw = 0, err_cw = 0, nat_cw = 0, bound_ratio = 0, tol_ratio = 0;
    int path = -1, reason = 0, blocks = 0, moduli = 0, log2_ratio = 0, level = 0;
    bool nan_ok = true;
};

ozd_handle_t g_h;

Result run(const Case& c, std::vector<double>* out_C = nullptr) {
    const int64_t m = c.m, n = c.n, k = c.k;
    const int64_t Ar = c.ta ? k : m, Ac = c.ta ? m : k;
    const int64_t Br = c.tb ? n : k, Bc = c.tb ? k : n;
    const int64_t lda = std::max<int64_t>(1, Ar + c.pad);
    const int64_t ldb = std::max<int64_t>(1, Br + c.pad);
    const int64_t ldc = std::max<int64_t>(1, m + c.pad);
    Gen g = Gen::parse(c.gen);
    bool inject_nan = false, inject_inf = false;
    if (g.kind == "nan") { g.kind = "uniform"; inject_nan = true; }
    if (g.kind == "inf") { g.kind = "uniform"; inject_inf = true; }
    bool zero_vecs = false;
    if (g.kind == "zerovecs") { g.kind = "uniform"; zero_vecs = true; }

    std::vector<double> A, B, C0;
    fill(A, Ar, Ac, lda, g, 1000 + m * 7 + k, !c.ta);
    fill(B, Br, Bc, ldb, g, 2000 + n * 11 + k, c.tb);
    fill(C0, m, n, ldc, Gen::parse("uniform"), 3000 + m + n, true);
    if (inject_nan) A[0] = NAN;
    if (inject_inf) B[(size_t)ldb * (Bc / 2)] = INFINITY;
    if (zero_vecs) {
        // zero every third row of op(A) and every fifth column of op(B)
        for (int64_t i = 0; i < m; i += 3)
            for (int64_t l = 0; l < k; ++l) (c.ta ? A[l + i * lda] : A[i + l * lda]) = 0;
        for (int64_t j = 0; j < n; j += 5)
            for (int64_t l = 0; l < k; ++l) (c.tb ? B[j + l * ldb] : B[l + j * ldb]) = 0;
    }

    double *dA, *dB, *dC, *dN, *dC0, *dR, *dAB;
    const size_t sA = A.size() * 8, sB = B.size() * 8, sC = C0.size() * 8;
    CK(cudaMalloc(&dA, sA)); CK(cudaMalloc(&dB, sB));
    CK(cudaMalloc(&dC, sC)); CK(cudaMalloc(&dN, sC)); CK(cudaMalloc(&dC0, sC));
    CK(cudaMalloc(&dR, sC)); CK(cudaMalloc(&dAB, sC));
    CK(cudaMemcpy(dA, A.data(), sA, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, B.data(), sB, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dC0, C0.data(), sC, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dC, dC0, sC, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(dN, dC0, sC, cudaMemcpyDeviceToDevice));

    const auto oa = c.ta ? OZD_OP_T : OZD_OP_N, ob = c.tb ? OZD_OP_T : OZD_OP_N;
    if (c.tol >= 0) ozd_set_accuracy(g_h, c.tol);
    else ozd_set_moduli(g_h, c.moduli);
    ozd_set_max_workspace(g_h, c.max_ws ? c.max_ws : size_t(4) << 30);
    ozd_set_backend(g_h, OZD_BACKEND_EMU);
    if (ozd_dgemm(g_h, oa, ob, m, n, k, c.alpha, dA, lda, dB, ldb, c.beta, dC, ldc) != OZD_OK) {
        std::fprintf(stderr, "ozd_dgemm failed\n");
        std::exit(2);
    }
    ozd_info_t info;
    ozd_get_info(g_h, &info);
    ozd_set_backend(g_h, OZD_BACKEND_NATIVE);
    ozd_dgemm(g_h, oa, ob, m, n, k, c.alpha, dA, lda, dB, ldb, c.beta, dN, ldc);
    ozd_set_backend(g_h, OZD_BACKEND_EMU);
    ref_gemm<<<dim3((unsigned)((m + 127) / 128), (unsigned)n), 128>>>(
        c.ta, c.tb, m, n, k, c.alpha, dA, lda, dB, ldb, c.beta, dC0, ldc, dR, dAB);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<double> C(C0.size()), N(C0.size()), R(C0.size()), AB(C0.size());
    CK(cudaMemcpy(C.data(), dC, sC, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(N.data(), dN, sC, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(R.data(), dR, sC, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(AB.data(), dAB, sC, cudaMemcpyDeviceToHost));
    for (auto p : {dA, dB, dC, dN, dC0, dR, dAB}) cudaFree(p);

    std::vector<double> nA, nB;
    vec_norms(A, Ar, Ac, lda, !c.ta, nA);
    vec_norms(B, Br, Bc, ldb, c.tb, nB);

    Result r;
    r.path = info.path;
    r.reason = info.fallback_reason;
    r.blocks = info.col_blocks;
    r.moduli = info.moduli;
    r.log2_ratio = info.log2_ratio;
    r.level = info.estimate_level;
    const double u = std::ldexp(1.0, -53);
    const double tol = c.tol == 0 ? k * u : c.tol;
    const int P = ozd_scale_bits(info.path == 1 ? info.moduli : 16);
    const double cfac = 4.0 * std::sqrt((double)k) * std::ldexp(1.0, -P);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i) {
            const size_t x = i + j * ldc;
            if (std::isnan(R[x]) || std::isinf(R[x])) {
                // non-finite reference: must match native exactly in class
                if (std::isnan(R[x]) != std::isnan(C[x]) || std::isinf(R[x]) != std::isinf(C[x]))
                    r.nan_ok = false;
                continue;
            }
            const double d = std::fabs(C[x] - R[x]);
            const double dn = std::fabs(N[x] - R[x]);
            const double nw = std::fabs(c.alpha) * nA[i] * nB[j];
            const double bound = cfac * nw + 4 * u * AB[x];
            if (!(d <= bound)) r.bound_ratio = std::max(r.bound_ratio, bound > 0 && !std::isnan(d) ? d / bound : INFINITY);
            else if (bound > 0) r.bound_ratio = std::max(r.bound_ratio, d / bound);
            if (nw > 0) r.err_nw = std::max(r.err_nw, d / nw);
            if (c.tol >= 0) {
                // AB = |alpha||A||B| + |beta C0|
                const double lim = tol * (AB[x] - std::fabs(c.beta * C0[x])) + 4 * u * AB[x];
                r.tol_ratio = std::max(r.tol_ratio, lim > 0 ? d / lim : (d > 0 ? INFINITY : 0.0));
            }
            if (AB[x] > 0) {
                r.err_cw = std::max(r.err_cw, d / AB[x]);
                r.nat_cw = std::max(r.nat_cw, dn / AB[x]);
            } else if (d != 0) {
                r.err_cw = INFINITY;
            }
        }
    if (out_C) *out_C = C;
    return r;
}

// OZD_BACKEND_AUTO: tiny problems and small-k HPL-like updates at 14 moduli
// must go native; a large square product with 8 moduli must be emulated.
// Timing-free: this checks the decision, which comes from the cost model.
int dispatch_checks() {
    struct D { const char* name; int64_t m, n, k; int moduli; int expect_path; };
    const D ds[] = {
        {"auto-tiny",         64,    64,    64, 14, 0},
        {"auto-skinny",    16384,    64, 16384, 14, 0},
        {"auto-hpl-k256",   8192,  8192,   256, 14, 0},
        {"auto-square-m8",  8192,  8192,  8192,  8, 1},
    };
    int fails = 0;
    ozd_set_backend(g_h, OZD_BACKEND_AUTO);
    ozd_set_max_workspace(g_h, size_t(4) << 30);  // earlier cases may have lowered it
    for (const auto& d : ds) {
        double *A, *B, *C;
        CK(cudaMalloc(&A, (size_t)d.m * d.k * 8));
        CK(cudaMalloc(&B, (size_t)d.k * d.n * 8));
        CK(cudaMalloc(&C, (size_t)d.m * d.n * 8));
        CK(cudaMemset(A, 0, (size_t)d.m * d.k * 8));  // zeros: fine for a dispatch check
        CK(cudaMemset(B, 0, (size_t)d.k * d.n * 8));
        ozd_set_moduli(g_h, d.moduli);
        ozd_dgemm(g_h, OZD_OP_N, OZD_OP_N, d.m, d.n, d.k, 1.0, A, d.m, B, d.k, 0.0, C, d.m);
        CK(cudaDeviceSynchronize());
        ozd_info_t info;
        ozd_get_info(g_h, &info);
        const bool ok = info.path == d.expect_path &&
                        (d.expect_path == 0 || info.fallback_reason == OZD_FALLBACK_NONE) &&
                        (d.expect_path == 1 || info.fallback_reason == OZD_FALLBACK_DISPATCH);
        fails += !ok;
        std::printf("%-18s %5ld %5ld %6ld pred native %8.3f ms emu %8.3f ms -> %s %s\n", d.name,
                    (long)d.m, (long)d.n, (long)d.k, info.pred_native_ms, info.pred_emu_ms,
                    info.path ? "emu" : "native", ok ? "PASS" : "FAIL");
        if (!ok) std::printf("  (fallback reason %d)\n", info.fallback_reason);
        cudaFree(A); cudaFree(B); cudaFree(C);
    }
    ozd_set_backend(g_h, OZD_BACKEND_EMU);
    return fails;
}

int adaptive_sweep() {
    std::printf("gen,m,n,k,tol,path,reason,moduli,log2_ratio,emu_cw,nat_cw,err_over_tol,level\n");
    for (const char* gen : {"uniform", "phi0.5", "phi1", "phi2", "phi4", "rowscale"})
        for (double tol : {0.0, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3})
            for (int64_t k : {256, 1024, 4096}) {
                Case c{"adapt", 1024, 1024, k, 0, 0, gen, 1.0, 0.0, 0, 0, 0, tol};
                Result r = run(c);
                std::printf("%s,%ld,%ld,%ld,%.0e,%d,%d,%d,%d,%.3e,%.3e,%.3f,%d\n", gen, (long)c.m,
                            (long)c.n, (long)k, tol, r.path, r.reason, r.moduli, r.log2_ratio,
                            r.err_cw, r.nat_cw, r.tol_ratio, r.level);
            }
    return 0;
}

int suite() {
    const Case cases[] = {
        // name                 m     n     k   ta tb gen          alpha beta  mod
        {"square-uniform",    512,  512,  512, 0, 0, "uniform",    1.0,  0.0, 14},
        {"square-alpha-beta", 512,  512,  512, 0, 0, "uniform",   -1.5,  0.75, 14},
        {"odd-shape",        1000,  777,  333, 0, 0, "uniform",    1.0,  1.0, 14},
        {"op-NT",             300,  200,  150, 0, 1, "uniform",    1.0,  0.0, 14},
        {"op-TN",             300,  200,  150, 1, 0, "uniform",    1.0,  0.0, 14},
        {"op-TT",             300,  200,  150, 1, 1, "uniform",    2.0,  -1.0, 14},
        {"padded-ld",         257,  129,   65, 0, 0, "uniform",    1.0,  0.5, 14, 13},
        {"1x1x1",               1,    1,    1, 0, 0, "uniform",    1.0,  0.0, 14},
        {"skinny-n",         4096,    8, 1024, 0, 0, "uniform",    1.0,  0.0, 14},
        {"small-k",          2048, 2048,   16, 0, 0, "uniform",    1.0, -1.0, 14},
        {"large-k",            64,   64, 16384, 0, 0, "uniform",   1.0,  0.0, 14},
        {"hpl-like",         2048, 2048, 1024, 0, 0, "uniform",   -1.0,  1.0, 14},
        {"phi0.5",            512,  512,  512, 0, 0, "phi0.5",     1.0,  0.0, 14},
        {"phi2",              512,  512,  512, 0, 0, "phi2",       1.0,  0.0, 14},
        {"phi4",              512,  512,  512, 0, 0, "phi4",       1.0,  0.0, 14},
        {"rowscale-2^300",    512,  512,  512, 0, 0, "rowscale",   1.0,  0.0, 14},
        {"zero-vectors",      300,  300,  300, 0, 0, "zerovecs",   1.0,  0.5, 14},
        {"moduli-8",          512,  512,  512, 0, 0, "uniform",    1.0,  0.0, 8},
        {"moduli-11",         512,  512,  512, 0, 0, "uniform",    1.0,  0.0, 11},
        {"moduli-16",         512,  512,  512, 0, 0, "phi2",       1.0,  0.0, 16},
        {"col-blocked",      1024, 1024,  512, 0, 0, "uniform",    1.0,  0.0, 14, 0, 24u << 20},
        {"nan-input",         128,  128,  128, 0, 0, "nan",        1.0,  0.0, 14},
        {"inf-input",         128,  128,  128, 0, 0, "inf",        1.0,  0.0, 14},
        // adaptive moduli: tol 0 = native DGEMM's worst-case bound k*2^-53
        {"adapt-uniform",    1024, 1024, 1024, 0, 0, "uniform",    1.0,  0.0,  0, 0, 0, 0.0, 1},
        {"adapt-hpl-beta",   2048, 2048, 1024, 0, 0, "uniform",   -1.0,  1.0,  0, 0, 0, 0.0},
        {"adapt-phi0.5",      512,  512,  512, 0, 0, "phi0.5",     1.0,  0.0,  0, 0, 0, 0.0},
        {"adapt-phi2",        512,  512,  512, 0, 0, "phi2",       1.0,  0.0,  0, 0, 0, 0.0},
        {"adapt-phi4",        512,  512,  512, 0, 0, "phi4",       1.0,  0.0,  0, 0, 0, 0.0},
        {"adapt-rowscale",    512,  512,  512, 0, 0, "rowscale",   1.0,  0.0,  0, 0, 0, 0.0},
        {"adapt-zerovecs",    300,  300,  300, 1, 1, "zerovecs",   1.0,  0.5,  0, 0, 0, 0.0},
        {"adapt-tol1e-6",    1024, 1024, 1024, 0, 0, "uniform",    1.0,  0.0,  0, 0, 0, 1e-6},
        {"adapt-tol1e-10",   1024, 1024, 1024, 0, 1, "phi2",       1.0,  0.0,  0, 0, 0, 1e-10},
        {"adapt-blocked",    1024, 1024,  512, 0, 0, "phi0.5",     1.0,  0.0,  0, 0, 24u << 20, 0.0},
        // level 2 (two-band |A||B| bound) for wide exponent ranges
        {"adapt2-phi2-1e-6", 1024, 1024, 1024, 0, 0, "phi2",       1.0,  0.0,  0, 0, 0, 1e-6, 2},
        {"adapt2-phi2-1e-8", 1024, 1024, 1024, 1, 1, "phi2",       1.0,  0.5,  0, 0, 0, 1e-8},
        {"adapt2-blocked",   1024, 1024,  512, 0, 0, "phi2",       1.0,  0.0,  0, 0, 24u << 20, 1e-6, 2},
    };
    int fails = 0;
    std::printf("%-18s %5s %5s %6s %3s %4s | %9s %9s %9s | %8s %8s %6s %s\n", "case", "m", "n",
                "k", "op", "mod", "emu_cw", "nat_cw", "emu_nw", "err/bnd", "err/tol", "path",
                "result");
    for (const auto& c : cases) {
        Result r = run(c);
        const bool non_finite = !std::strcmp(c.gen, "nan") || !std::strcmp(c.gen, "inf");
        bool ok;
        if (non_finite) ok = r.path == 0 && r.reason == OZD_FALLBACK_NONFINITE && r.nan_ok;
        else if (c.tol >= 0)  // guarantee met, or declared unreachable
            ok = r.nan_ok && ((r.path == 1 && r.bound_ratio <= 1.0 && r.tol_ratio <= 1.0) ||
                              (r.path == 0 && r.reason == OZD_FALLBACK_ACCURACY));
        else ok = r.path == 1 && r.bound_ratio <= 1.0 && r.nan_ok;
        if (c.max_ws) ok = ok && r.blocks > 1;
        if (c.expect_level >= 0) ok = ok && r.path == 1 && r.level == c.expect_level;
        fails += !ok;
        char op[3] = {c.ta ? 'T' : 'N', c.tb ? 'T' : 'N', 0};
        char path[16];
        if (r.path == 1) std::snprintf(path, sizeof path, r.level ? "emu/L%d" : "emu", r.level);
        else std::snprintf(path, sizeof path, "nat/%d", r.reason);
        std::printf("%-18s %5ld %5ld %6ld %3s %4d | %9.2e %9.2e %9.2e | %8.3f %8.3f %6s %s\n",
                    c.name, (long)c.m, (long)c.n, (long)c.k, op, r.moduli, r.err_cw, r.nat_cw,
                    r.err_nw, r.bound_ratio, r.tol_ratio, path, ok ? "PASS" : "FAIL");
    }
    fails += dispatch_checks();

    // Column blocking must not change the answer, and repeated calls must be
    // bitwise identical.
    {
        Case a{"blk", 1024, 1024, 512, 0, 0, "phi2", 1.0, 0.0, 14};
        Case b = a;
        b.max_ws = 24u << 20;
        std::vector<double> c1, c2, c3;
        run(a, &c1);
        run(a, &c3);
        Result rb = run(b, &c2);
        const bool same_blk = c1 == c2 && rb.blocks > 1;
        const bool same_rep = c1 == c3;
        std::printf("%-18s bitwise equal to unblocked (%d blocks): %s\n", "blocking-invariance",
                    rb.blocks, same_blk ? "PASS" : "FAIL");
        std::printf("%-18s bitwise equal on repeat: %s\n", "determinism", same_rep ? "PASS" : "FAIL");
        fails += !same_blk + !same_rep;
    }
    std::printf("\n%d failure(s)\n", fails);
    return fails ? 1 : 0;
}

int sweep() {
    std::printf("gen,m,n,k,moduli,P,emu_cw,emu_nw,nat_cw\n");
    for (const char* gen : {"uniform", "phi0.5", "phi2", "phi4"})
        for (int mod = 4; mod <= 16; ++mod) {
            Case c{"sweep", 1024, 1024, 1024, 0, 0, gen, 1.0, 0.0, mod};
            Result r = run(c);
            std::printf("%s,%ld,%ld,%ld,%d,%d,%.3e,%.3e,%.3e\n", gen, (long)c.m, (long)c.n,
                        (long)c.k, mod, ozd_scale_bits(mod), r.err_cw, r.err_nw, r.nat_cw);
        }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (ozd_create(&g_h, 0) != OZD_OK) return 2;
    const char* mode = argc > 1 ? argv[1] : "";
    int rc = !std::strcmp(mode, "--sweep")      ? sweep()
             : !std::strcmp(mode, "--adaptive") ? adaptive_sweep()
                                                : suite();
    ozd_destroy(g_h);
    return rc;
}
