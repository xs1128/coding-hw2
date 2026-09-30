// INT8 Ozaki scheme II DGEMM emulation.
//
// Pipeline per call (see README for the math):
//   1. scale   : per row of op(A) / column of op(B), a power-of-two exponent
//                so that the scaled integer vectors have 2-norm < 2^P.
//   2. convert : A' = trunc(A * 2^e), residues A' mod m_t as int8, one slab per
//                modulus, K padded to a multiple of 16 (TN layout for IMMA).
//   3. gemm    : one strided-batched cuBLAS INT8 GEMM (INT32 accumulate)
//                covering all moduli.
//   4. crt     : reduce each INT32 product mod m_t, CRT-reconstruct the
//                integer X in (-M/4, M/4), unscale,
//                C = alpha * X * 2^-(eA+eB) + beta * C.
// v0 (tag v0-baseline) ran a separate INT32 -> uint8 reduction pass and one
// GEMM launch per modulus.
// Columns of C are processed in blocks so the workspace stays under a cap.
#include "ozdgemm.h"

#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {

constexpr int kMaxModuli = 16;
constexpr int kMaxChunks = 4;    // 40-bit chunks of M (M < 2^126)
constexpr int kChunkBits = 40;
constexpr int kPadK = 16;
constexpr int64_t kMaxK = 131072 - kPadK;  // keeps |sum| of 128*128 products < 2^31

// Pairwise coprime moduli, all <= 256 so residues fit in int8.
constexpr int kModuli[kMaxModuli] = {256, 255, 253, 251, 247, 241, 239, 233,
                                     229, 227, 223, 217, 211, 199, 197, 193};

struct CrtConst {
    int    n;                         // number of moduli
    int    nch;                       // number of 40-bit chunks of M
    int    P;                         // scaled vectors have 2-norm < 2^P
    int    mod[kMaxModuli];
    double S[kMaxModuli][kMaxChunks]; // chunks of M_t * (M_t^-1 mod m_t)
    double Mc[kMaxChunks];            // chunks of M
    double pow40[kMaxChunks];         // 2^(40 q)
    double invM;
    double md[kMaxModuli];            // m_t as double
    double inv_m[kMaxModuli];         // 1 / m_t
    double c32[kMaxModuli];           // 2^32 mod m_t
    double half[kMaxModuli];          // largest residue kept positive
    uint32_t magic[kMaxModuli];       // u / m_t = see udiv() (Granlund-Montgomery)
    int      shift[kMaxModuli];
    uint32_t r31[kMaxModuli];         // 2^31 mod m_t
};

__constant__ CrtConst c_crt;

using u128 = unsigned __int128;

int64_t modinv(int64_t a, int64_t m) {
    int64_t t = 0, nt = 1, r = m, nr = a % m;
    while (nr) {
        int64_t q = r / nr;
        int64_t tmp = t - q * nt; t = nt; nt = tmp;
        tmp = r - q * nr; r = nr; nr = tmp;
    }
    return t < 0 ? t + m : t;
}

int floor_log2(u128 x) {
    int b = -1;
    while (x) { x >>= 1; ++b; }
    return b;
}

CrtConst make_crt(int n) {
    CrtConst c{};
    c.n = n;
    u128 M = 1;
    for (int t = 0; t < n; ++t) { c.mod[t] = kModuli[t]; M *= (u128)kModuli[t]; }
    const int lg = floor_log2(M);
    c.P = (lg - 2) / 2;  // 2^(2P) <= M/4
    c.nch = lg / kChunkBits + 1;
    const u128 mask = ((u128)1 << kChunkBits) - 1;
    double Md = 0;
    for (int q = 0; q < c.nch; ++q) {
        c.Mc[q] = (double)(uint64_t)((M >> (kChunkBits * q)) & mask);
        Md += std::ldexp(c.Mc[q], kChunkBits * q);
    }
    c.invM = 1.0 / Md;
    for (int q = 0; q < kMaxChunks; ++q) c.pow40[q] = std::ldexp(1.0, kChunkBits * q);
    for (int t = 0; t < n; ++t) {
        const int mt = kModuli[t];
        c.md[t] = mt;
        c.inv_m[t] = 1.0 / mt;
        c.c32[t] = (double)((uint64_t(1) << 32) % mt);
        c.half[t] = mt == 256 ? 127 : (mt - 1) / 2;
        int l = 0;
        while ((1 << l) < mt) ++l;  // ceil(log2 m_t)
        c.shift[t] = l;
        c.magic[t] = (uint32_t)(((uint64_t(1) << 32) * ((uint64_t(1) << l) - mt)) / mt + 1);
        c.r31[t] = (uint32_t)((uint64_t(1) << 31) % mt);
    }
    for (int t = 0; t < n; ++t) {
        const u128 Mt = M / (u128)kModuli[t];
        const int64_t y = modinv((int64_t)(Mt % (u128)kModuli[t]), kModuli[t]);
        const u128 St = Mt * (u128)y;  // < M
        for (int q = 0; q < c.nch; ++q)
            c.S[t][q] = (double)(uint64_t)((St >> (kChunkBits * q)) & mask);
    }
    return c;
}

int64_t round_up(int64_t x, int64_t a) { return (x + a - 1) / a * a; }

// ---------------------------------------------------------------- kernels

// x mod m in [0, m) for an integer-valued double with |x| < 2^52.
__device__ __forceinline__ double mod_small(double x, double m, double inv_m) {
    const double q = floor(x * inv_m);  // may be off by one; fixed below
    double r = fma(-q, m, x);           // exact
    r += r < 0.0 ? m : 0.0;
    r -= r >= m ? m : 0.0;
    return r;
}

// Running (exponent, scaled sum of squares) for an overflow-free 2-norm:
// norm^2 = ss * 4^s, with s = ilogb(max |x|) seen so far.
struct Nrm {
    int s;      // -100000 while empty
    double ss;
};
constexpr Nrm kNrmEmpty = {-100000, 0.0};

__device__ __forceinline__ void nrm_add(Nrm& n, double a) {
    if (a == 0.0) return;
    const int ea = ilogb(a);
    if (ea > n.s) { n.ss = ldexp(n.ss, 2 * (n.s - ea)); n.s = ea; }
    const double t = ldexp(a, -n.s);
    n.ss = fma(t, t, n.ss);
}

__device__ __forceinline__ void nrm_merge(Nrm& n, const Nrm& o) {
    const int m = max(n.s, o.s);
    n.ss = ldexp(n.ss, 2 * (n.s - m)) + ldexp(o.ss, 2 * (o.s - m));
    n.s = m;
}

// Per-vector exponents computed by the scale kernels:
//   norm : floor(log2 ||x||_2) (rounded up slightly), so ||x|| < 2^(norm+1)
//   amax : ilogb(max |x|)
// Zero vectors get kZeroVec for both. The scaling exponent for P bits is
// e = P - 1 - norm, so ||x * 2^e|| < 2^P.
constexpr int kZeroVec = -(1 << 20);

struct VecExp {
    int norm, amax;
};

__device__ VecExp vec_exp(const Nrm& n) {
    if (n.ss == 0.0) return {kZeroVec, kZeroVec};
    const double nrm = sqrt(n.ss) * (1.0 + 0x1p-40);
    return {ilogb(nrm) + n.s, n.s};
}

__device__ __forceinline__ int scale_exp(int norm) { return c_crt.P - 1 - norm; }

// Vectors with contiguous elements (stride_l == 1): one 128-thread block each.
__global__ void scale_contig(const double* __restrict__ X, int64_t len, int64_t sv,
                             VecExp* __restrict__ e, int* __restrict__ flag) {
    const int64_t v = blockIdx.x;
    const double* x = X + v * sv;
    Nrm n = kNrmEmpty;
    bool bad = false;
    for (int64_t l = threadIdx.x; l < len; l += blockDim.x) {
        const double a = x[l];
        bad |= !isfinite(a);
        if (!bad) nrm_add(n, fabs(a));
    }
    for (int o = 16; o; o >>= 1) {
        const Nrm p = {__shfl_xor_sync(~0u, n.s, o), __shfl_xor_sync(~0u, n.ss, o)};
        nrm_merge(n, p);
    }
    __shared__ Nrm part[4];
    __shared__ int any_bad;
    if (threadIdx.x == 0) any_bad = 0;
    __syncthreads();
    if (bad) any_bad = 1;
    if (threadIdx.x % 32 == 0) part[threadIdx.x / 32] = n;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int)blockDim.x / 32; ++w) nrm_merge(n, part[w]);
        if (any_bad) atomicExch(flag, 1);
        e[v] = vec_exp(n);
    }
}

// Vectors with strided elements (stride_v == 1): a 32x32 block covers 32
// vectors; threadIdx.x walks vectors (coalesced), threadIdx.y splits the length.
__global__ void scale_strided(const double* __restrict__ X, int64_t nvec, int64_t len,
                              int64_t sl, VecExp* __restrict__ e, int* __restrict__ flag) {
    const int64_t v = (int64_t)blockIdx.x * 32 + threadIdx.x;
    Nrm n = kNrmEmpty;
    bool bad = false;
    if (v < nvec)
        for (int64_t l = threadIdx.y; l < len; l += 32) {
            const double a = X[v + l * sl];
            bad |= !isfinite(a);
            if (!bad) nrm_add(n, fabs(a));
        }
    __shared__ Nrm part[32][33];
    part[threadIdx.y][threadIdx.x] = n;
    if (bad) atomicExch(flag, 1);
    __syncthreads();
    if (threadIdx.y == 0 && v < nvec) {
        for (int y = 1; y < 32; ++y) nrm_merge(n, part[y][threadIdx.x]);
        e[v] = vec_exp(n);
    }
}

// Writes the symmetric int8 residues of 4 consecutive integer-valued doubles
// (|val| < 2^61) for every modulus: o[t*slab + u] = val[u] mod m_t.
__device__ __forceinline__ void write_residues(const double* val4, int8_t* o, int64_t slab) {
    // |val| < 2^61: split as hi*2^32 + lo with lo in [0, 2^32) so that every
    // mod_small argument stays below 2^52.
    double hi[4], lo[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const double val = val4[u];
        hi[u] = floor(val * 0x1p-32);
        lo[u] = fma(-hi[u], 0x1p32, val);
    }
    const int n = c_crt.n;
    for (int t = 0; t < n; ++t) {
        const double m = c_crt.md[t], im = c_crt.inv_m[t], c32 = c_crt.c32[t];
        const double half = c_crt.half[t];
        uint32_t packed = 0;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            double rr = mod_small(fma(mod_small(hi[u], m, im), c32, lo[u]), m, im);
            if (rr > half) rr -= m;
            packed |= (uint32_t)(uint8_t)(int8_t)(int)rr << (8 * u);
        }
        *reinterpret_cast<uint32_t*>(o + t * slab) = packed;
    }
}

// Element (v, l) of a set of vectors lives at X[v*sv + l*sl]. Writes, for each
// modulus t, out[t*slab + v*kp + l] = trunc(x * 2^e[v]) mod m_t (symmetric
// int8), zero-padding l in [len, kp). A 32x32 tile goes through shared memory
// so the read is coalesced for either stride; each thread then writes 4
// consecutive residues as one 32-bit store.
//
// The |x| modes quantize s = |x| * 2^(6 - amax[v]), s in [0, 128), for the
// |A||B| lower bound that picks the moduli count:
//   MODE 1 (level 1): one slab of h = floor(s);
//   MODE 2 (level 2): h, and in the next slab l = floor((s - h) * 128).
// h, s - h and l are exact, all in [0, 127], and s >= h + l/128, so INT8 GEMMs
// of these slabs give lower bounds on |A||B|.
template <int MODE>
__global__ void convert(const double* __restrict__ X, int64_t sv, int64_t sl,
                        int64_t v0, int64_t nvec, int64_t len, int64_t kp,
                        const VecExp* __restrict__ e, int8_t* __restrict__ out,
                        int64_t slab) {
    __shared__ double tile[32][33];
    const int64_t vb = (int64_t)blockIdx.y * 32, lb = (int64_t)blockIdx.x * 32;
    const bool contig_l = (sl == 1);
    for (int r = threadIdx.y; r < 32; r += blockDim.y) {
        const int vi = contig_l ? r : threadIdx.x;
        const int li = contig_l ? threadIdx.x : r;
        const int64_t v = vb + vi, l = lb + li;
        double val = 0.0;
        if (v < nvec && l < len) {
            const double a = X[(v0 + v) * sv + l * sl];
            const VecExp ev = e[v0 + v];
            val = MODE ? ldexp(fabs(a), 6 - ev.amax) : trunc(ldexp(a, scale_exp(ev.norm)));
        }
        tile[vi][li] = val;
    }
    __syncthreads();
    const int tid = threadIdx.y * 32 + threadIdx.x;  // blockDim = (32, 8)
    const int r = tid >> 3, lg = (tid & 7) * 4;
    const int64_t v = vb + r, l = lb + lg;
    if (v >= nvec || l >= kp) return;  // kp % 16 == 0, so l+3 < kp too
    if constexpr (MODE > 0) {
        uint32_t hi = 0, lo = 0;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const double x = tile[r][lg + u], h = floor(x);
            hi |= (uint32_t)h << (8 * u);
            lo |= (uint32_t)floor((x - h) * 128.0) << (8 * u);
        }
        *reinterpret_cast<uint32_t*>(out + v * kp + l) = hi;
        if (MODE == 2) *reinterpret_cast<uint32_t*>(out + slab + v * kp + l) = lo;
    } else {
        write_residues(&tile[r][lg], out + v * kp + l, slab);
    }
}

// floor(u / m_t) for any 32-bit u (Granlund & Montgomery 1994, fig. 4.1).
__device__ __forceinline__ uint32_t udiv(uint32_t u, int t) {
    const uint32_t t1 = __umulhi(c_crt.magic[t], u);
    return (t1 + ((u - t1) >> 1)) >> (c_crt.shift[t] - 1);
}

// x mod m_t in [0, m_t) for a signed 32-bit x, integer ALU only.
__device__ __forceinline__ uint32_t imod(int32_t x, int t) {
    const uint32_t u = (uint32_t)x ^ 0x80000000u;  // x + 2^31
    const uint32_t m = c_crt.mod[t];
    int32_t r = (int32_t)(u - udiv(u, t) * m) - (int32_t)c_crt.r31[t];
    return (uint32_t)(r < 0 ? r + (int32_t)m : r);
}

// Exact conversion of a small unsigned int to double without I2F.
__device__ __forceinline__ double u2d(uint32_t v) {
    return __hiloint2double(0x43300000, (int)v) - 0x1p52;
}

__device__ __forceinline__ void two_sum(double a, double b, double& s, double& err) {
    s = a + b;
    const double bb = s - a;
    err = (a - (s - bb)) + (b - bb);
}

// 2^k as a double, exact for normal k; ldexp otherwise.
__device__ __forceinline__ double scale2(double x, int k) {
    if (k >= -1022 && k <= 1023) return x * __longlong_as_double((long long)(k + 1023) << 52);
    return ldexp(x, k);
}

// Mod reduction of the per-modulus INT32 products + CRT reconstruction for an
// m x nb block of C starting at column j0. C32 holds one mp x nb slab per
// modulus. NCH = number of 40-bit chunks of M.
template <int NCH>
__global__ void crt(const int32_t* __restrict__ C32, int64_t slab, int64_t mp,
                    int64_t m, int64_t nb, int64_t j0, const VecExp* __restrict__ eA,
                    const VecExp* __restrict__ eB, double alpha, double beta,
                    double* __restrict__ C, int64_t ldc) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const int n = c_crt.n;
    const int ea = scale_exp(eA[i].norm);
    for (int64_t j = blockIdx.y; j < nb; j += gridDim.y) {
        const int32_t* p = C32 + i + j * mp;
        double s[NCH];
#pragma unroll
        for (int q = 0; q < NCH; ++q) s[q] = 0.0;
        for (int t = 0; t < n; ++t) {
            const double c = u2d(imod(p[t * slab], t));
#pragma unroll
            for (int q = 0; q < NCH; ++q) s[q] = fma(c, c_crt.S[t][q], s[q]);  // exact: < 2^52
        }
        double approx = 0.0;
#pragma unroll
        for (int q = NCH - 1; q >= 0; --q) approx = fma(approx, 0x1p40, s[q]);
        const double qm = rint(approx * c_crt.invM);
        double hi = 0.0, lo = 0.0;
#pragma unroll
        for (int q = NCH - 1; q >= 0; --q) {
            // exact: both terms are integers below 2^52, times a power of two
            const double r = fma(-qm, c_crt.Mc[q], s[q]) * c_crt.pow40[q];
            double err;
            two_sum(hi, r, hi, err);
            lo += err;
        }
        const double x = scale2(hi + lo, -(ea + scale_exp(eB[j0 + j].norm)));
        double* c = C + i + (j0 + j) * ldc;
        *c = beta == 0.0 ? alpha * x : fma(alpha, x, beta * *c);
    }
}

// Computes the max over entries of an upper bound on
//   log2( ||A_i|| * ||B_j|| / (|A||B|)_ij ),
// using ||A_i|| < 2^(norm+1) and a lower bound on (|A||B|)_ij from the INT8
// GEMMs of the quantized |A|, |B| slabs (convert MODE 1/2):
//   LEVEL 1: G = hh,                          |A||B| >= G * 2^(amaxA + amaxB - 12)
//   LEVEL 2: G = hh*2^14 + (hl + lh)*2^7 + ll, |A||B| >= G * 2^(amaxA + amaxB - 26)
// with the four products in consecutive slabs of G (hh, hl, lh, ll), summed in
// 64 bits because hl + lh alone can exceed INT32 near kMaxK.
// Entries with a zero row/column need no accuracy (their result is exact);
// entries with G_ij == 0 get kNoBound (the bound is unusable there).
constexpr int kNoBound = 1 << 20;

template <int LEVEL>
__global__ void ratio_bound(const int32_t* __restrict__ G, int64_t slab, int64_t mp, int64_t m,
                            int64_t nb, int64_t j0, const VecExp* __restrict__ eA,
                            const VecExp* __restrict__ eB, int* __restrict__ out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int best = -kNoBound;
    if (i < m) {
        const VecExp a = eA[i];
        for (int64_t j = blockIdx.y; j < nb; j += gridDim.y) {
            const VecExp b = eB[j0 + j];
            if (a.norm == kZeroVec || b.norm == kZeroVec) continue;
            const int64_t o = i + j * mp;
            uint64_t g = (uint32_t)G[o];
            if (LEVEL == 2)
                g = (g << 14) + (((uint64_t)(uint32_t)G[o + slab] + (uint32_t)G[o + 2 * slab]) << 7) +
                    (uint32_t)G[o + 3 * slab];
            const int shift = LEVEL == 2 ? 28 : 14;
            const int t = g > 0 ? a.norm + b.norm + shift - a.amax - b.amax - (63 - __clzll(g))
                                : kNoBound;
            best = max(best, t);
        }
    }
    for (int o = 16; o; o >>= 1) best = max(best, __shfl_xor_sync(~0u, best, o));
    if (threadIdx.x % 32 == 0) atomicMax(out, best);
}

// Copies the two flags to mapped host memory with plain stores. A
// cudaMemcpyAsync would go through the device-to-host copy engine and queue
// behind any large copy the caller has in flight on another stream.
__global__ void publish_flags(const int* __restrict__ flag, int* __restrict__ host) {
    if (threadIdx.x < 2) host[threadIdx.x] = flag[threadIdx.x];
}

}  // namespace

// ---------------------------------------------------------------- handle

struct ozd_context {
    cudaStream_t      stream = nullptr;
    cublasHandle_t    cublas = nullptr;
    ozd_backend_t     backend = OZD_BACKEND_EMU;
    int               moduli = 14;
    double            tol = -1.0;  // < 0: fixed moduli; 0: DGEMM-level; > 0: target
    CrtConst          crt{};
    ozd_cost_model_t  model{};
    size_t            max_ws = size_t(4) << 30;
    bool              safety = true;
    bool              profiling = false;
    void*             ws = nullptr;
    size_t            ws_bytes = 0;
    int*              h_flag = nullptr;  // mapped pinned: [0] non-finite flag, [1] log2 ratio bound
    int*              d_hflag = nullptr; // device alias of h_flag
    cudaEvent_t       ev[6] = {};
    ozd_info_t        info{};
};

namespace {

#define OZD_CUDA(x)                                                              \
    do {                                                                         \
        cudaError_t e_ = (x);                                                    \
        if (e_ != cudaSuccess) {                                                 \
            std::fprintf(stderr, "ozdgemm: %s:%d %s\n", __FILE__, __LINE__,       \
                         cudaGetErrorString(e_));                                \
            return OZD_ERR_CUDA;                                                 \
        }                                                                        \
    } while (0)

#define OZD_CUBLAS(x)                                                            \
    do {                                                                         \
        cublasStatus_t s_ = (x);                                                 \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                       \
            std::fprintf(stderr, "ozdgemm: %s:%d cublas status %d\n", __FILE__,   \
                         __LINE__, (int)s_);                                     \
            return OZD_ERR_CUBLAS;                                               \
        }                                                                        \
    } while (0)

// Defaults fitted on an H200 from job 441237 (scripts/fit_model.py): worst
// prediction error 17%, 63/63 dispatch decisions correct on that grid. The
// CRT read traffic moves in lockstep with the INT32 GEMM output, so the fit
// folds it into gemm_out_gbs and crt_gbs is effectively infinite.
constexpr ozd_cost_model_t kDefaultModel = {
    /*native_overhead_ms*/ 0.01265, /*native_tflops*/ 56.66,
    /*emu_overhead_ms*/ 0.05899, /*int8_tops*/ 1182.0, /*convert_gbs*/ 949.4,
    /*gemm_out_gbs*/ 1637.0, /*crt_gbs*/ 1e30,
};

// Scaling bits P for each moduli count (index = count).
int scale_bits_table(int n) {
    static int tab[kMaxModuli + 1] = {0};
    if (!tab[kMaxModuli])
        for (int i = 2; i <= kMaxModuli; ++i) tab[i] = make_crt(i).P;
    return tab[n];
}

// Smallest moduli count whose scaling gives at least P bits, or 0 if none.
int moduli_for_bits(int P) {
    for (int n = 2; n <= kMaxModuli; ++n)
        if (scale_bits_table(n) >= P) return n;
    return 0;
}

// Bits needed so that 4*sqrt(k)*2^-P * 2^log2_ratio <= tol (see README).
int bits_needed(int log2_ratio, int64_t k, double tol) {
    return (int)std::ceil(log2_ratio + 2.0 + 0.5 * std::log2((double)k) - std::log2(tol));
}

double pred_native_ms(const ozd_cost_model_t& c, double m, double n, double k) {
    return c.native_overhead_ms + 2 * m * n * k / (c.native_tflops * 1e9);
}

// Emulated time for N moduli; with estimate=true, add the |A||B| bound pass.
double pred_emu_ms(const ozd_cost_model_t& c, double m, double n, double kp, int N,
                   bool beta_nonzero, bool estimate) {
    const double conv = (m + n) * kp * (8 + N) / (c.convert_gbs * 1e6);
    const double gemm1 = 2 * m * n * kp / (c.int8_tops * 1e9) + 4 * m * n / (c.gemm_out_gbs * 1e6);
    const double crt = (4.0 * N + 8 + (beta_nonzero ? 8 : 0)) * m * n / (c.crt_gbs * 1e6);
    double t = c.emu_overhead_ms + conv + N * gemm1 + crt;
    if (estimate)
        t += (m + n) * kp * 9 / (c.convert_gbs * 1e6) + gemm1 + 4 * m * n / (c.crt_gbs * 1e6);
    return t;
}

// Level-2 bound pass: two |x| slabs per operand, four INT8 GEMMs, 16 bytes
// of INT32 products read per output entry.
double pred_level2_ms(const ozd_cost_model_t& c, double m, double n, double kp) {
    const double gemm1 = 2 * m * n * kp / (c.int8_tops * 1e9) + 4 * m * n / (c.gemm_out_gbs * 1e6);
    return (m + n) * kp * 10 / (c.convert_gbs * 1e6) + 4 * gemm1 + 16 * m * n / (c.crt_gbs * 1e6);
}

struct Layout {
    int64_t kp, mp, nb;
    size_t  off_eA, off_eB, off_flag, off_A8, off_B8, off_C32, total;
};

size_t align256(size_t x) { return (x + 255) & ~size_t(255); }

// Column block width nb and workspace offsets for a given problem.
bool plan(int nmod, int64_t m, int64_t n, int64_t k, size_t cap, Layout& L) {
    L.kp = round_up(std::max<int64_t>(k, 1), kPadK);
    L.mp = round_up(m, 4);
    const size_t fixed = align256(8 * m) + align256(8 * n) + 256 +
                         align256((size_t)nmod * m * L.kp) + 4 * 256;
    // per column of C: converted B for every modulus + one INT32 product per modulus
    const size_t per_col = (size_t)nmod * L.kp + (size_t)4 * nmod * L.mp;
    if (fixed + per_col > cap) return false;
    L.nb = std::min<int64_t>(n, (int64_t)((cap - fixed) / per_col));
    size_t o = 0;
    L.off_eA = o;   o += align256(8 * m);
    L.off_eB = o;   o += align256(8 * n);
    L.off_flag = o; o += 256;
    L.off_A8 = o;   o += align256((size_t)nmod * m * L.kp);
    L.off_B8 = o;   o += align256((size_t)nmod * L.nb * L.kp);
    L.off_C32 = o;  o += align256((size_t)4 * nmod * L.mp * L.nb);
    L.total = o;
    return true;
}

int plan_moduli(ozd_handle_t h) { return h->tol >= 0 ? kMaxModuli : h->moduli; }

ozd_status_t ensure_ws(ozd_handle_t h, size_t bytes) {
    if (bytes <= h->ws_bytes) return OZD_OK;
    if (h->ws) OZD_CUDA(cudaFree(h->ws));
    h->ws = nullptr;
    h->ws_bytes = 0;
    OZD_CUDA(cudaMalloc(&h->ws, bytes));
    h->ws_bytes = bytes;
    return OZD_OK;
}

// c_crt is process-global, so track what is loaded globally rather than per
// handle (handles with different moduli counts may be interleaved).
int g_loaded_moduli = 0;

ozd_status_t load_moduli(ozd_handle_t h, int n) {
    if (h->crt.n != n) h->crt = make_crt(n);
    if (g_loaded_moduli == n) return OZD_OK;
    OZD_CUDA(cudaMemcpyToSymbolAsync(c_crt, &h->crt, sizeof(CrtConst), 0,
                                     cudaMemcpyHostToDevice, h->stream));
    g_loaded_moduli = n;
    return OZD_OK;
}

cublasOperation_t cop(ozd_op_t o) { return o == OZD_OP_N ? CUBLAS_OP_N : CUBLAS_OP_T; }

ozd_status_t native(ozd_handle_t h, ozd_op_t ta, ozd_op_t tb, int64_t m, int64_t n,
                    int64_t k, double alpha, const double* A, int64_t lda,
                    const double* B, int64_t ldb, double beta, double* C,
                    int64_t ldc, int reason) {
    OZD_CUBLAS(cublasDgemm_64(h->cublas, cop(ta), cop(tb), m, n, k, &alpha, A, lda,
                              B, ldb, &beta, C, ldc));
    h->info.path = 0;
    h->info.moduli = 0;
    h->info.fallback_reason = reason;
    return OZD_OK;
}

struct StageTimer {
    ozd_handle_t h;
    int cur = 0;
    void mark() {
        if (h->profiling) cudaEventRecord(h->ev[cur++ % 2], h->stream);
    }
    // Elapsed since the previous mark, recorded now.
    double lap() {
        if (!h->profiling) return 0.0;
        cudaEvent_t a = h->ev[(cur + 1) % 2], b = h->ev[cur % 2];
        cudaEventRecord(b, h->stream);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        ++cur;
        return ms;
    }
};

dim3 convert_grid(int64_t kp, int64_t nvec) {
    return dim3((unsigned)((kp + 31) / 32), (unsigned)((nvec + 31) / 32));
}

}  // namespace

// ---------------------------------------------------------------- API

extern "C" {

ozd_status_t ozd_create(ozd_handle_t* out, cudaStream_t stream) {
    if (!out) return OZD_ERR_ARG;
    auto* h = new ozd_context;
    h->stream = stream;
    h->model = kDefaultModel;
    if (cublasCreate(&h->cublas) != CUBLAS_STATUS_SUCCESS) { delete h; return OZD_ERR_CUBLAS; }
    cublasSetStream(h->cublas, stream);
    OZD_CUDA(cudaHostAlloc(&h->h_flag, 2 * sizeof(int), cudaHostAllocMapped));
    OZD_CUDA(cudaHostGetDevicePointer(&h->d_hflag, h->h_flag, 0));
    for (auto& e : h->ev) OZD_CUDA(cudaEventCreate(&e));
    *out = h;
    return OZD_OK;
}

ozd_status_t ozd_destroy(ozd_handle_t h) {
    if (!h) return OZD_OK;
    if (h->ws) cudaFree(h->ws);
    if (h->h_flag) cudaFreeHost(h->h_flag);
    for (auto& e : h->ev) if (e) cudaEventDestroy(e);
    if (h->cublas) cublasDestroy(h->cublas);
    delete h;
    return OZD_OK;
}

ozd_status_t ozd_set_backend(ozd_handle_t h, ozd_backend_t b) {
    if (b != OZD_BACKEND_NATIVE && b != OZD_BACKEND_EMU && b != OZD_BACKEND_AUTO) return OZD_ERR_ARG;
    h->backend = b;
    return OZD_OK;
}

ozd_status_t ozd_set_moduli(ozd_handle_t h, int moduli) {
    if (moduli < 2 || moduli > kMaxModuli) return OZD_ERR_ARG;
    h->moduli = moduli;
    h->tol = -1.0;
    return OZD_OK;
}

ozd_status_t ozd_set_accuracy(ozd_handle_t h, double tol) {
    if (!(tol >= 0.0) || tol >= 1.0) return OZD_ERR_ARG;
    h->tol = tol;
    return OZD_OK;
}

ozd_status_t ozd_set_cost_model(ozd_handle_t h, const ozd_cost_model_t* m) {
    if (!h || !m) return OZD_ERR_ARG;
    h->model = *m;
    return OZD_OK;
}

ozd_status_t ozd_get_cost_model(ozd_handle_t h, ozd_cost_model_t* m) {
    if (!h || !m) return OZD_ERR_ARG;
    *m = h->model;
    return OZD_OK;
}

ozd_status_t ozd_set_max_workspace(ozd_handle_t h, size_t bytes) { h->max_ws = bytes; return OZD_OK; }
ozd_status_t ozd_set_safety_check(ozd_handle_t h, int on) { h->safety = on != 0; return OZD_OK; }
ozd_status_t ozd_set_profiling(ozd_handle_t h, int on) { h->profiling = on != 0; return OZD_OK; }
cublasHandle_t ozd_cublas(ozd_handle_t h) { return h->cublas; }
const char* ozd_version(void) { return "v2"; }
int ozd_scale_bits(int moduli) {
    return moduli < 2 || moduli > kMaxModuli ? -1 : scale_bits_table(moduli);
}

size_t ozd_workspace_bytes(ozd_handle_t h, int64_t m, int64_t n, int64_t k) {
    Layout L;
    return plan(plan_moduli(h), m, n, k, h->max_ws, L) ? L.total : 0;
}

ozd_status_t ozd_reserve(ozd_handle_t h, int64_t m, int64_t n, int64_t k) {
    const size_t b = ozd_workspace_bytes(h, m, n, k);
    if (!b) return OZD_ERR_WORKSPACE;
    return ensure_ws(h, b);
}

ozd_status_t ozd_get_info(ozd_handle_t h, ozd_info_t* info) {
    if (!h || !info) return OZD_ERR_ARG;
    *info = h->info;
    return OZD_OK;
}

ozd_status_t ozd_dgemm(ozd_handle_t h, ozd_op_t ta, ozd_op_t tb, int64_t m,
                       int64_t n, int64_t k, double alpha, const double* A,
                       int64_t lda, const double* B, int64_t ldb, double beta,
                       double* C, int64_t ldc) {
    if (!h || m < 0 || n < 0 || k < 0) return OZD_ERR_ARG;
    if (ldc < std::max<int64_t>(1, m)) return OZD_ERR_ARG;
    std::memset(&h->info, 0, sizeof h->info);
    if (m == 0 || n == 0) return OZD_OK;
    auto fallback = [&](int reason) {
        return native(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, reason);
    };
    if (h->backend == OZD_BACKEND_NATIVE) return fallback(0);
    if (k == 0 || alpha == 0.0 || k > kMaxK) return fallback(OZD_FALLBACK_DEGENERATE);

    // Moduli count: fixed, or (adaptive) at least what the tolerance needs
    // even in the best case log2_ratio = 0; refined after the bound pass.
    const bool adaptive = h->tol >= 0;
    const double tol = h->tol == 0 ? std::ldexp((double)k, -53) : h->tol;
    const int nmod_min = adaptive ? moduli_for_bits(bits_needed(0, k, tol)) : h->moduli;
    int nmod = nmod_min;
    if (nmod == 0) return fallback(OZD_FALLBACK_ACCURACY);

    const int64_t kp = round_up(k, kPadK);
    const bool auto_bk = h->backend == OZD_BACKEND_AUTO;
    h->info.pred_native_ms = pred_native_ms(h->model, m, n, k);
    h->info.pred_emu_ms = pred_emu_ms(h->model, m, n, kp, nmod, beta != 0, adaptive);
    if (auto_bk && h->info.pred_emu_ms >= h->info.pred_native_ms)
        return fallback(OZD_FALLBACK_DISPATCH);

    Layout L;
    if (!plan(plan_moduli(h), m, n, k, h->max_ws, L)) return fallback(OZD_FALLBACK_DEGENERATE);
    if (auto s = ensure_ws(h, L.total)) return s;

    char* ws = (char*)h->ws;
    VecExp* eA = (VecExp*)(ws + L.off_eA);
    VecExp* eB = (VecExp*)(ws + L.off_eB);
    int* flag = (int*)(ws + L.off_flag);  // [0] non-finite, [1] log2 ratio bound
    int8_t* A8 = (int8_t*)(ws + L.off_A8);
    int8_t* B8 = (int8_t*)(ws + L.off_B8);
    int32_t* C32 = (int32_t*)(ws + L.off_C32);
    cudaStream_t st = h->stream;
    StageTimer tm{h};
    const dim3 cblk(32, 8);
    const int32_t one = 1, zero = 0;

    // Row i of op(A): element l at A[i*svA + l*slA]; column j of op(B) likewise.
    const int64_t svA = ta == OZD_OP_N ? 1 : lda, slA = ta == OZD_OP_N ? lda : 1;
    const int64_t svB = tb == OZD_OP_N ? ldb : 1, slB = tb == OZD_OP_N ? 1 : ldb;

    nvtxRangePushA("ozd:scale");
    tm.mark();
    OZD_CUDA(cudaMemsetAsync(flag, 0, sizeof(int), st));
    OZD_CUDA(cudaMemsetAsync(flag + 1, 0x80, sizeof(int), st));  // very negative
    auto scale = [&](const double* X, int64_t nvec, int64_t sv, int64_t sl, VecExp* e) {
        if (sl == 1)
            scale_contig<<<(unsigned)nvec, 128, 0, st>>>(X, k, sv, e, flag);
        else
            scale_strided<<<(unsigned)((nvec + 31) / 32), dim3(32, 32), 0, st>>>(X, nvec, k, sl, e, flag);
    };
    scale(A, m, svA, slA, eA);
    scale(B, n, svB, slB, eB);
    OZD_CUDA(cudaGetLastError());
    nvtxRangePop();
    h->info.t_scale_ms += tm.lap();

    if (adaptive) {
        // Lower bound of |A||B| via one INT8 GEMM of floor-quantized |A|, |B|.
        nvtxRangePushA("ozd:estimate");
        convert<1><<<convert_grid(kp, m), cblk, 0, st>>>(A, svA, slA, 0, m, k, kp, eA, A8, 0);
        for (int64_t j0 = 0; j0 < n; j0 += L.nb) {
            const int64_t nb = std::min(L.nb, n - j0);
            convert<1><<<convert_grid(kp, nb), cblk, 0, st>>>(B, svB, slB, j0, nb, k, kp, eB, B8, 0);
            OZD_CUBLAS(cublasGemmEx_64(h->cublas, CUBLAS_OP_T, CUBLAS_OP_N, m, nb, kp, &one,
                                       A8, CUDA_R_8I, kp, B8, CUDA_R_8I, kp, &zero, C32,
                                       CUDA_R_32I, L.mp, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
            const dim3 g((unsigned)((m + 255) / 256), (unsigned)std::min<int64_t>(nb, 4096));
            ratio_bound<1><<<g, 256, 0, st>>>(C32, 0, L.mp, m, nb, j0, eA, eB, flag + 1);
        }
        OZD_CUDA(cudaGetLastError());
        nvtxRangePop();
        h->info.t_estimate_ms += tm.lap();
    }

    if (h->safety || adaptive) {
        publish_flags<<<1, 32, 0, st>>>(flag, h->d_hflag);
        OZD_CUDA(cudaGetLastError());
        OZD_CUDA(cudaStreamSynchronize(st));
        if (h->h_flag[0]) return fallback(OZD_FALLBACK_NONFINITE);
    }
    if (adaptive) {
        // The ratio is >= 1 by Cauchy-Schwarz, so clamp the bound at 0.
        int lr = std::max(h->h_flag[1], 0);
        h->info.estimate_level = 1;
        nmod = lr >= kNoBound ? 0 : moduli_for_bits(bits_needed(lr, k, tol));
        if (nmod == 0) {
            // Level 1 cannot certify any count (wide exponent range: some
            // lower bounds round to 0, or are too loose). Try the finer
            // two-band bound, unless AUTO predicts native wins even with the
            // fewest moduli the tolerance allows.
            if (auto_bk && pred_emu_ms(h->model, m, n, kp, nmod_min, beta != 0, false) +
                                   pred_level2_ms(h->model, m, n, kp) >=
                               h->info.pred_native_ms) {
                h->info.log2_ratio = lr;
                return fallback(OZD_FALLBACK_DISPATCH);
            }
            nvtxRangePushA("ozd:estimate2");
            OZD_CUDA(cudaMemsetAsync(flag + 1, 0x80, sizeof(int), st));
            const int64_t sA = m * kp;
            convert<2><<<convert_grid(kp, m), cblk, 0, st>>>(A, svA, slA, 0, m, k, kp, eA, A8, sA);
            for (int64_t j0 = 0; j0 < n; j0 += L.nb) {
                const int64_t nb = std::min(L.nb, n - j0);
                const int64_t sB = nb * kp, sC = L.mp * nb;
                convert<2><<<convert_grid(kp, nb), cblk, 0, st>>>(B, svB, slB, j0, nb, k, kp, eB, B8, sB);
                // C32 slabs: hh, hl, lh, ll
                const int8_t* xa[4] = {A8, A8, A8 + sA, A8 + sA};
                const int8_t* xb[4] = {B8, B8 + sB, B8, B8 + sB};
                for (int q = 0; q < 4; ++q)
                    OZD_CUBLAS(cublasGemmEx_64(h->cublas, CUBLAS_OP_T, CUBLAS_OP_N, m, nb, kp, &one,
                                               xa[q], CUDA_R_8I, kp, xb[q], CUDA_R_8I, kp, &zero,
                                               C32 + q * sC, CUDA_R_32I, L.mp, CUBLAS_COMPUTE_32I,
                                               CUBLAS_GEMM_DEFAULT));
                const dim3 g((unsigned)((m + 255) / 256), (unsigned)std::min<int64_t>(nb, 4096));
                ratio_bound<2><<<g, 256, 0, st>>>(C32, sC, L.mp, m, nb, j0, eA, eB, flag + 1);
            }
            OZD_CUDA(cudaGetLastError());
            publish_flags<<<1, 32, 0, st>>>(flag, h->d_hflag);
            OZD_CUDA(cudaGetLastError());
            OZD_CUDA(cudaStreamSynchronize(st));
            nvtxRangePop();
            h->info.t_estimate_ms += tm.lap();
            lr = std::max(h->h_flag[1], 0);
            h->info.estimate_level = 2;
            nmod = lr >= kNoBound ? 0 : moduli_for_bits(bits_needed(lr, k, tol));
        }
        h->info.log2_ratio = lr;
        if (nmod == 0) return fallback(OZD_FALLBACK_ACCURACY);
        h->info.pred_emu_ms = pred_emu_ms(h->model, m, n, kp, nmod, beta != 0, false);
        if (auto_bk && h->info.pred_emu_ms >= h->info.pred_native_ms)
            return fallback(OZD_FALLBACK_DISPATCH);
    }
    if (auto s = load_moduli(h, nmod)) return s;

    nvtxRangePushA("ozd:convertA");
    const int64_t slabA = m * kp;
    convert<0><<<convert_grid(kp, m), cblk, 0, st>>>(A, svA, slA, 0, m, k, kp, eA, A8, slabA);
    OZD_CUDA(cudaGetLastError());
    nvtxRangePop();
    h->info.t_convert_ms += tm.lap();

    int blocks = 0;
    for (int64_t j0 = 0; j0 < n; j0 += L.nb, ++blocks) {
        const int64_t nb = std::min(L.nb, n - j0);
        const int64_t slabB = nb * kp, slabC = L.mp * nb;

        nvtxRangePushA("ozd:convertB");
        convert<0><<<convert_grid(kp, nb), cblk, 0, st>>>(B, svB, slB, j0, nb, k, kp, eB, B8, slabB);
        OZD_CUDA(cudaGetLastError());
        nvtxRangePop();
        h->info.t_convert_ms += tm.lap();

        nvtxRangePushA("ozd:gemm");
        OZD_CUBLAS(cublasGemmStridedBatchedEx_64(
            h->cublas, CUBLAS_OP_T, CUBLAS_OP_N, m, nb, kp, &one, A8, CUDA_R_8I, kp,
            slabA, B8, CUDA_R_8I, kp, slabB, &zero, C32, CUDA_R_32I, L.mp, slabC, nmod,
            CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
        nvtxRangePop();
        h->info.t_gemm_ms += tm.lap();

        nvtxRangePushA("ozd:crt");
        const dim3 cgrid((unsigned)((m + 255) / 256), (unsigned)std::min<int64_t>(nb, 65535));
        switch (h->crt.nch) {
            case 1: crt<1><<<cgrid, 256, 0, st>>>(C32, slabC, L.mp, m, nb, j0, eA, eB, alpha, beta, C, ldc); break;
            case 2: crt<2><<<cgrid, 256, 0, st>>>(C32, slabC, L.mp, m, nb, j0, eA, eB, alpha, beta, C, ldc); break;
            case 3: crt<3><<<cgrid, 256, 0, st>>>(C32, slabC, L.mp, m, nb, j0, eA, eB, alpha, beta, C, ldc); break;
            default: crt<4><<<cgrid, 256, 0, st>>>(C32, slabC, L.mp, m, nb, j0, eA, eB, alpha, beta, C, ldc); break;
        }
        OZD_CUDA(cudaGetLastError());
        nvtxRangePop();
        h->info.t_crt_ms += tm.lap();
    }
    h->info.path = 1;
    h->info.moduli = nmod;
    h->info.col_blocks = blocks;
    return OZD_OK;
}

}  // extern "C"
