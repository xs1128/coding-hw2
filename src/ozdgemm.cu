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

// Exponent e such that ||x * 2^e||_2 < 2^P.
__device__ int vec_exponent(const Nrm& n, int P) {
    if (n.ss == 0.0) return 0;
    const double nrm = sqrt(n.ss) * (1.0 + 0x1p-40);
    return P - 1 - (ilogb(nrm) + n.s);
}

// Vectors with contiguous elements (stride_l == 1): one 128-thread block each.
__global__ void scale_contig(const double* __restrict__ X, int64_t len, int64_t sv,
                             int* __restrict__ e, int* __restrict__ flag) {
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
        e[v] = vec_exponent(n, c_crt.P);
    }
}

// Vectors with strided elements (stride_v == 1): a 32x32 block covers 32
// vectors; threadIdx.x walks vectors (coalesced), threadIdx.y splits the length.
__global__ void scale_strided(const double* __restrict__ X, int64_t nvec, int64_t len,
                              int64_t sl, int* __restrict__ e, int* __restrict__ flag) {
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
        e[v] = vec_exponent(n, c_crt.P);
    }
}

// Element (v, l) of a set of vectors lives at X[v*sv + l*sl]. Writes, for each
// modulus t, out[t*slab + v*kp + l] = trunc(x * 2^e[v]) mod m_t (symmetric
// int8), zero-padding l in [len, kp). A 32x32 tile goes through shared memory
// so the read is coalesced for either stride; each thread then writes 4
// consecutive residues as one 32-bit store.
__global__ void convert(const double* __restrict__ X, int64_t sv, int64_t sl,
                        int64_t v0, int64_t nvec, int64_t len, int64_t kp,
                        const int* __restrict__ e, int8_t* __restrict__ out,
                        int64_t slab) {
    __shared__ double tile[32][33];
    const int64_t vb = (int64_t)blockIdx.y * 32, lb = (int64_t)blockIdx.x * 32;
    const bool contig_l = (sl == 1);
    for (int r = threadIdx.y; r < 32; r += blockDim.y) {
        const int vi = contig_l ? r : threadIdx.x;
        const int li = contig_l ? threadIdx.x : r;
        const int64_t v = vb + vi, l = lb + li;
        double val = 0.0;
        if (v < nvec && l < len) val = trunc(ldexp(X[(v0 + v) * sv + l * sl], e[v0 + v]));
        tile[vi][li] = val;
    }
    __syncthreads();
    const int tid = threadIdx.y * 32 + threadIdx.x;  // blockDim = (32, 8)
    const int r = tid >> 3, lg = (tid & 7) * 4;
    const int64_t v = vb + r, l = lb + lg;
    if (v >= nvec || l >= kp) return;  // kp % 16 == 0, so l+3 < kp too
    // |val| < 2^61: split as hi*2^32 + lo with lo in [0, 2^32) so that every
    // mod_small argument stays below 2^52.
    double hi[4], lo[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const double val = tile[r][lg + u];
        hi[u] = floor(val * 0x1p-32);
        lo[u] = fma(-hi[u], 0x1p32, val);
    }
    int8_t* o = out + v * kp + l;
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
                    int64_t m, int64_t nb, int64_t j0, const int* __restrict__ eA,
                    const int* __restrict__ eB, double alpha, double beta,
                    double* __restrict__ C, int64_t ldc) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const int n = c_crt.n;
    const int ea = eA[i];
    for (int64_t j = blockIdx.y; j < nb; j += gridDim.y) {
        const int32_t* p = C32 + i + j * mp;
        double s[NCH];
#pragma unroll
        for (int q = 0; q < NCH; ++q) s[q] = 0.0;
        for (int t = 0; t < n; ++t) {
            const double c = mod_small((double)p[t * slab], c_crt.md[t], c_crt.inv_m[t]);
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
        const double x = scale2(hi + lo, -(ea + eB[j0 + j]));
        double* c = C + i + (j0 + j) * ldc;
        *c = beta == 0.0 ? alpha * x : fma(alpha, x, beta * *c);
    }
}

}  // namespace

// ---------------------------------------------------------------- handle

struct ozd_context {
    cudaStream_t   stream = nullptr;
    cublasHandle_t cublas = nullptr;
    ozd_backend_t  backend = OZD_BACKEND_EMU;
    int            moduli = 14;
    CrtConst       crt{};
    size_t         max_ws = size_t(4) << 30;
    bool           safety = true;
    bool           profiling = false;
    void*          ws = nullptr;
    size_t         ws_bytes = 0;
    int*           h_flag = nullptr;  // pinned
    cudaEvent_t    ev[6] = {};
    ozd_info_t     info{};
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

struct Layout {
    int64_t kp, mp, nb;
    size_t  off_eA, off_eB, off_flag, off_A8, off_B8, off_C32, total;
};

size_t align256(size_t x) { return (x + 255) & ~size_t(255); }

// Column block width nb and workspace offsets for a given problem.
bool plan(int nmod, int64_t m, int64_t n, int64_t k, size_t cap, Layout& L) {
    L.kp = round_up(std::max<int64_t>(k, 1), kPadK);
    L.mp = round_up(m, 4);
    const size_t fixed = align256(4 * m) + align256(4 * n) + 256 +
                         align256((size_t)nmod * m * L.kp) + 4 * 256;
    // per column of C: converted B for every modulus + one INT32 product per modulus
    const size_t per_col = (size_t)nmod * L.kp + (size_t)4 * nmod * L.mp;
    if (fixed + per_col > cap) return false;
    L.nb = std::min<int64_t>(n, (int64_t)((cap - fixed) / per_col));
    size_t o = 0;
    L.off_eA = o;   o += align256(4 * m);
    L.off_eB = o;   o += align256(4 * n);
    L.off_flag = o; o += 256;
    L.off_A8 = o;   o += align256((size_t)nmod * m * L.kp);
    L.off_B8 = o;   o += align256((size_t)nmod * L.nb * L.kp);
    L.off_C32 = o;  o += align256((size_t)4 * nmod * L.mp * L.nb);
    L.total = o;
    return true;
}

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

ozd_status_t load_moduli(ozd_handle_t h) {
    if (h->crt.n != h->moduli) h->crt = make_crt(h->moduli);
    if (g_loaded_moduli == h->moduli) return OZD_OK;
    OZD_CUDA(cudaMemcpyToSymbolAsync(c_crt, &h->crt, sizeof(CrtConst), 0,
                                     cudaMemcpyHostToDevice, h->stream));
    g_loaded_moduli = h->moduli;
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

}  // namespace

// ---------------------------------------------------------------- API

extern "C" {

ozd_status_t ozd_create(ozd_handle_t* out, cudaStream_t stream) {
    if (!out) return OZD_ERR_ARG;
    auto* h = new ozd_context;
    h->stream = stream;
    if (cublasCreate(&h->cublas) != CUBLAS_STATUS_SUCCESS) { delete h; return OZD_ERR_CUBLAS; }
    cublasSetStream(h->cublas, stream);
    OZD_CUDA(cudaMallocHost(&h->h_flag, sizeof(int)));
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
    if (b != OZD_BACKEND_NATIVE && b != OZD_BACKEND_EMU) return OZD_ERR_ARG;
    h->backend = b;
    return OZD_OK;
}

ozd_status_t ozd_set_moduli(ozd_handle_t h, int moduli) {
    if (moduli < 2 || moduli > kMaxModuli) return OZD_ERR_ARG;
    h->moduli = moduli;
    return OZD_OK;
}

ozd_status_t ozd_set_max_workspace(ozd_handle_t h, size_t bytes) { h->max_ws = bytes; return OZD_OK; }
ozd_status_t ozd_set_safety_check(ozd_handle_t h, int on) { h->safety = on != 0; return OZD_OK; }
ozd_status_t ozd_set_profiling(ozd_handle_t h, int on) { h->profiling = on != 0; return OZD_OK; }
cublasHandle_t ozd_cublas(ozd_handle_t h) { return h->cublas; }
const char* ozd_version(void) { return "v1"; }
int ozd_scale_bits(int moduli) {
    return moduli < 2 || moduli > kMaxModuli ? -1 : make_crt(moduli).P;
}

size_t ozd_workspace_bytes(ozd_handle_t h, int64_t m, int64_t n, int64_t k) {
    Layout L;
    return plan(h->moduli, m, n, k, h->max_ws, L) ? L.total : 0;
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
    if (h->backend == OZD_BACKEND_NATIVE)
        return native(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 0);
    if (k == 0 || alpha == 0.0 || k > kMaxK)
        return native(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 2);

    Layout L;
    if (!plan(h->moduli, m, n, k, h->max_ws, L))
        return native(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 2);
    if (auto s = ensure_ws(h, L.total)) return s;
    if (auto s = load_moduli(h)) return s;

    char* ws = (char*)h->ws;
    int* eA = (int*)(ws + L.off_eA);
    int* eB = (int*)(ws + L.off_eB);
    int* flag = (int*)(ws + L.off_flag);
    int8_t* A8 = (int8_t*)(ws + L.off_A8);
    int8_t* B8 = (int8_t*)(ws + L.off_B8);
    int32_t* C32 = (int32_t*)(ws + L.off_C32);
    const int nmod = h->moduli;
    cudaStream_t st = h->stream;
    StageTimer tm{h};

    // Row i of op(A): element l at A[i*svA + l*slA]; column j of op(B) likewise.
    const int64_t svA = ta == OZD_OP_N ? 1 : lda, slA = ta == OZD_OP_N ? lda : 1;
    const int64_t svB = tb == OZD_OP_N ? ldb : 1, slB = tb == OZD_OP_N ? 1 : ldb;

    nvtxRangePushA("ozd:scale");
    tm.mark();
    OZD_CUDA(cudaMemsetAsync(flag, 0, sizeof(int), st));
    auto scale = [&](const double* X, int64_t nvec, int64_t sv, int64_t sl, int* e) {
        if (sl == 1)
            scale_contig<<<(unsigned)nvec, 128, 0, st>>>(X, k, sv, e, flag);
        else
            scale_strided<<<(unsigned)((nvec + 31) / 32), dim3(32, 32), 0, st>>>(X, nvec, k, sl, e, flag);
    };
    scale(A, m, svA, slA, eA);
    scale(B, n, svB, slB, eB);
    OZD_CUDA(cudaGetLastError());
    nvtxRangePop();
    if (h->safety) {
        OZD_CUDA(cudaMemcpyAsync(h->h_flag, flag, sizeof(int), cudaMemcpyDeviceToHost, st));
        OZD_CUDA(cudaStreamSynchronize(st));
        if (*h->h_flag)
            return native(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1);
    }
    h->info.t_scale_ms += tm.lap();

    nvtxRangePushA("ozd:convertA");
    const dim3 cblk(32, 8);
    const int64_t slabA = m * L.kp;
    convert<<<dim3((unsigned)(L.kp / 32 + (L.kp % 32 != 0)), (unsigned)((m + 31) / 32)), cblk, 0, st>>>(
        A, svA, slA, 0, m, k, L.kp, eA, A8, slabA);
    OZD_CUDA(cudaGetLastError());
    nvtxRangePop();
    h->info.t_convert_ms += tm.lap();

    const int32_t one = 1, zero = 0;
    int blocks = 0;
    for (int64_t j0 = 0; j0 < n; j0 += L.nb, ++blocks) {
        const int64_t nb = std::min(L.nb, n - j0);
        const int64_t slabB = nb * L.kp, slabC = L.mp * nb;

        nvtxRangePushA("ozd:convertB");
        convert<<<dim3((unsigned)(L.kp / 32 + (L.kp % 32 != 0)), (unsigned)((nb + 31) / 32)), cblk, 0, st>>>(
            B, svB, slB, j0, nb, k, L.kp, eB, B8, slabB);
        OZD_CUDA(cudaGetLastError());
        nvtxRangePop();
        h->info.t_convert_ms += tm.lap();

        nvtxRangePushA("ozd:gemm");
        OZD_CUBLAS(cublasGemmStridedBatchedEx_64(
            h->cublas, CUBLAS_OP_T, CUBLAS_OP_N, m, nb, L.kp, &one, A8, CUDA_R_8I, L.kp,
            slabA, B8, CUDA_R_8I, L.kp, slabB, &zero, C32, CUDA_R_32I, L.mp, slabC, nmod,
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
