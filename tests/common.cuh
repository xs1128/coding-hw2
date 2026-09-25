// Shared helpers for tests and benchmarks: input generators and a
// double-double reference GEMM.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define CK(x)                                                                    \
    do {                                                                         \
        cudaError_t e_ = (x);                                                    \
        if (e_ != cudaSuccess) {                                                 \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,               \
                         cudaGetErrorString(e_));                                \
            std::exit(2);                                                        \
        }                                                                        \
    } while (0)

// Input distributions.
//   uniform  : U(-0.5, 0.5), the HPL generator's distribution
//   phi<x>   : (U(0,1) - 0.5) * exp(x * N(0,1)), the Ozaki-paper test family;
//              larger x = wider exponent range
//   rowscale : uniform, then each vector scaled by 2^s, s in [-300, 300]
struct Gen {
    std::string kind;
    double phi = 0.0;
    static Gen parse(const std::string& s) {
        Gen g;
        if (s.rfind("phi", 0) == 0) { g.kind = "phi"; g.phi = std::atof(s.c_str() + 3); }
        else g.kind = s;
        return g;
    }
};

// Fills a rows x cols column-major matrix with leading dimension ld.
// vec_is_row says whether the "vectors" for rowscale are rows (A) or columns (B).
inline void fill(std::vector<double>& X, int64_t rows, int64_t cols, int64_t ld,
                 const Gen& g, uint64_t seed, bool vec_is_row) {
    X.assign((size_t)ld * cols, 0.0);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> Nd(0.0, 1.0);
    std::vector<int> sc;
    if (g.kind == "rowscale") {
        std::uniform_int_distribution<int> S(-300, 300);
        sc.resize(vec_is_row ? rows : cols);
        for (auto& s : sc) s = S(rng);
    }
    for (int64_t j = 0; j < cols; ++j)
        for (int64_t i = 0; i < rows; ++i) {
            double v;
            if (g.kind == "phi") v = (U(rng) - 0.5) * std::exp(g.phi * Nd(rng));
            else v = U(rng) - 0.5;
            if (!sc.empty()) v = std::ldexp(v, sc[vec_is_row ? i : j]);
            X[i + j * ld] = v;
        }
}

// ---------------------------------------------------------------- reference

struct dd { double hi, lo; };

// Non-finite values propagate like plain FP64 (the error terms would turn
// Inf into NaN).
__device__ inline dd dd_add(dd a, double b) {
    double s = a.hi + b, bb = s - a.hi;
    if (!isfinite(s)) return {s, 0.0};
    double e = (a.hi - (s - bb)) + (b - bb) + a.lo;
    double h = s + e;
    return {h, e - (h - s)};
}
__device__ inline dd dd_add_prod(dd a, double x, double y) {
    double p = x * y;
    if (!isfinite(p)) return dd_add(a, p);
    double pe = fma(x, y, -p);
    dd t = dd_add(a, p);
    return dd_add(t, pe);
}

// R = alpha*op(A)*op(B) + beta*C0 in double-double (rounded to double), and
// AB = alpha|op(A)||op(B)| + |beta C0| for componentwise error.
__global__ void ref_gemm(int ta, int tb, int64_t m, int64_t n, int64_t k, double alpha,
                         const double* A, int64_t lda, const double* B, int64_t ldb,
                         double beta, const double* C0, int64_t ldc, double* R,
                         double* AB) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t j = blockIdx.y;
    if (i >= m || j >= n) return;
    dd acc{0, 0};
    double abs_acc = 0;
    for (int64_t l = 0; l < k; ++l) {
        double a = ta ? A[l + i * lda] : A[i + l * lda];
        double b = tb ? B[j + l * ldb] : B[l + j * ldb];
        acc = dd_add_prod(acc, a, b);
        abs_acc = fma(fabs(a), fabs(b), abs_acc);
    }
    // alpha*acc + beta*c0, still in double-double
    double c0 = C0[i + j * ldc];
    dd r = {acc.hi * alpha, fma(acc.hi, alpha, -acc.hi * alpha) + acc.lo * alpha};
    r = dd_add_prod(r, beta, c0);
    R[i + j * ldc] = r.hi + r.lo;
    AB[i + j * ldc] = fabs(alpha) * abs_acc + fabs(beta * c0);
}

inline void vec_norms(const std::vector<double>& X, int64_t rows, int64_t cols, int64_t ld,
                      bool rows_are_vecs, std::vector<double>& out) {
    out.assign(rows_are_vecs ? rows : cols, 0.0);
    for (int64_t j = 0; j < cols; ++j)
        for (int64_t i = 0; i < rows; ++i) {
            double v = X[i + j * ld];
            out[rows_are_vecs ? i : j] += v * v;
        }
    for (auto& v : out) v = std::sqrt(v);
}
