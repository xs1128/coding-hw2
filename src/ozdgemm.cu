// INT8 Ozaki scheme II DGEMM emulation, v0 (unfused baseline).
//
// Pipeline per call (see README for the math):
//   1. scale   : per row of op(A) / column of op(B), a power-of-two exponent
//                so that the scaled integer vectors have 2-norm < 2^P.
//   2. convert : A' = trunc(A * 2^e), residues A' mod m_t as int8, one slab per
//                modulus, K padded to a multiple of 16 (TN layout for IMMA).
//   3. gemm    : one cuBLAS INT8 GEMM (INT32 accumulate) per modulus.
//   4. reduce  : INT32 result mod m_t -> uint8.
//   5. crt     : CRT reconstruction to the integer X in (-M/4, M/4), unscale,
//                C = alpha * X * 2^-(eA+eB) + beta * C.
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
    double invM;
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

// Exponent for one vector given its max-abs and scaled sum of squares.
__device__ int vec_exponent(double amax, double sumsq, int P) {
    if (amax == 0.0) return 0;
    const int s = ilogb(amax);
    const double nrm = sqrt(sumsq) * (1.0 + 0x1p-40);
    return P - 1 - (ilogb(nrm) + s);
}

// Vectors whose elements are contiguous (stride_l == 1): one warp per vector.
__global__ void scale_contig(const double* __restrict__ X, int64_t nvec,
                             int64_t len, int64_t sv, int* __restrict__ e,
                             int* __restrict__ flag) {
    const int64_t v = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (v >= nvec) return;
    const double* x = X + v * sv;
    double amax = 0.0;
    bool bad = false;
    for (int64_t l = lane; l < len; l += 32) {
        const double a = x[l];
        bad |= !isfinite(a);
        amax = fmax(amax, fabs(a));
    }
    for (int o = 16; o; o >>= 1) amax = fmax(amax, __shfl_xor_sync(~0u, amax, o));
    double ss = 0.0;
    if (amax > 0.0) {
        const int s = ilogb(amax);
        for (int64_t l = lane; l < len; l += 32) {
            const double a = ldexp(x[l], -s);
            ss = fma(a, a, ss);
        }
    }
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(~0u, ss, o);
    const unsigned anybad = __any_sync(~0u, bad);
    if (lane == 0) {
        if (anybad) atomicExch(flag, 1);
        e[v] = vec_exponent(amax, ss, c_crt.P);
    }
}

// Vectors that are strided (stride_v == 1, stride_l = ld): one thread per vector,
// neighbouring threads read neighbouring addresses.
__global__ void scale_strided(const double* __restrict__ X, int64_t nvec,
                              int64_t len, int64_t sl, int* __restrict__ e,
                              int* __restrict__ flag) {
    const int64_t v = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nvec) return;
    double amax = 0.0;
    bool bad = false;
    for (int64_t l = 0; l < len; ++l) {
        const double a = X[v + l * sl];
        bad |= !isfinite(a);
        amax = fmax(amax, fabs(a));
    }
    double ss = 0.0;
    if (amax > 0.0) {
        const int s = ilogb(amax);
        for (int64_t l = 0; l < len; ++l) {
            const double a = ldexp(X[v + l * sl], -s);
            ss = fma(a, a, ss);
        }
    }
    if (bad) atomicExch(flag, 1);
    e[v] = vec_exponent(amax, ss, c_crt.P);
}

__device__ __forceinline__ int8_t residue(long long x, int m) {
    if (m == 256) return (int8_t)(x & 0xFF);
    int r = (int)(x % m);
    const int h = (m - 1) >> 1;
    if (r > h) r -= m;
    if (r < -h) r += m;
    return (int8_t)r;
}

// Element (v, l) of a set of vectors lives at X[v*sv + l*sl]. Writes, for each
// modulus t, out[t*slab + v*kp + l] = trunc(x * 2^e[v]) mod m_t, zero-padding
// l in [len, kp). Tiled 32x32 so both the read and the write are coalesced.
__global__ void convert(const double* __restrict__ X, int64_t sv, int64_t sl,
                        int64_t v0, int64_t nvec, int64_t len, int64_t kp,
                        const int* __restrict__ e, int8_t* __restrict__ out,
                        int64_t slab) {
    __shared__ long long tile[32][33];
    const int64_t vb = (int64_t)blockIdx.y * 32, lb = (int64_t)blockIdx.x * 32;
    const bool contig_l = (sl == 1);
    for (int r = threadIdx.y; r < 32; r += blockDim.y) {
        // contig_l: threadIdx.x walks l; otherwise it walks v.
        const int vi = contig_l ? r : threadIdx.x;
        const int li = contig_l ? threadIdx.x : r;
        const int64_t v = vb + vi, l = lb + li;
        long long val = 0;
        if (v < nvec && l < len) {
            const double a = X[(v0 + v) * sv + l * sl];
            val = __double2ll_rz(ldexp(a, e[v0 + v]));
        }
        tile[vi][li] = val;
    }
    __syncthreads();
    const int n = c_crt.n;
    for (int r = threadIdx.y; r < 32; r += blockDim.y) {
        const int64_t v = vb + r, l = lb + threadIdx.x;
        if (v >= nvec || l >= kp) continue;
        const long long val = tile[r][threadIdx.x];
        int8_t* o = out + v * kp + l;
        for (int t = 0; t < n; ++t) o[t * slab] = residue(val, c_crt.mod[t]);
    }
}

// INT32 GEMM result mod m -> uint8 in [0, m). count is a multiple of 4.
__global__ void reduce_mod(const int4* __restrict__ in, uchar4* __restrict__ out,
                           int64_t count4, int m) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count4) return;
    const int4 v = in[i];
    auto f = [m](int x) -> unsigned char {
        if (m == 256) return (unsigned char)(x & 0xFF);
        int r = x % m;
        return (unsigned char)(r < 0 ? r + m : r);
    };
    out[i] = make_uchar4(f(v.x), f(v.y), f(v.z), f(v.w));
}

__device__ __forceinline__ void two_sum(double a, double b, double& s, double& err) {
    s = a + b;
    const double bb = s - a;
    err = (a - (s - bb)) + (b - bb);
}

// CRT reconstruction for an m x nb block of C starting at column j0.
__global__ void crt(const unsigned char* __restrict__ U, int64_t slab, int64_t mp,
                    int64_t m, int64_t nb, int64_t j0, const int* __restrict__ eA,
                    const int* __restrict__ eB, double alpha, double beta,
                    double* __restrict__ C, int64_t ldc) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i = idx % mp, j = idx / mp;
    if (i >= m || j >= nb) return;
    const int n = c_crt.n, nch = c_crt.nch;
    double s[kMaxChunks] = {0, 0, 0, 0};
    for (int t = 0; t < n; ++t) {
        const double c = (double)U[t * slab + idx];
#pragma unroll
        for (int q = 0; q < kMaxChunks; ++q)
            if (q < nch) s[q] = fma(c, c_crt.S[t][q], s[q]);  // exact: < 2^52
    }
    double approx = 0.0;
    for (int q = nch - 1; q >= 0; --q) approx = fma(approx, 0x1p40, s[q]);
    const double qm = rint(approx * c_crt.invM);
    double hi = 0.0, lo = 0.0;
    for (int q = nch - 1; q >= 0; --q) {
        const double r = ldexp(fma(-qm, c_crt.Mc[q], s[q]), kChunkBits * q);  // exact
        double e;
        two_sum(hi, r, hi, e);
        lo += e;
    }
    const double x = ldexp(hi + lo, -(eA[i] + eB[j0 + j]));
    double* c = C + i + (j0 + j) * ldc;
    *c = beta == 0.0 ? alpha * x : fma(alpha, x, beta * *c);
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
    size_t  off_eA, off_eB, off_flag, off_A8, off_B8, off_C32, off_U8, total;
};

size_t align256(size_t x) { return (x + 255) & ~size_t(255); }

// Column block width nb and workspace offsets for a given problem.
bool plan(int nmod, int64_t m, int64_t n, int64_t k, size_t cap, Layout& L) {
    L.kp = round_up(std::max<int64_t>(k, 1), kPadK);
    L.mp = round_up(m, 4);
    const size_t fixed = align256(4 * m) + align256(4 * n) + 256 +
                         align256((size_t)nmod * m * L.kp) + 4 * 256;
    const size_t per_col = (size_t)nmod * L.kp + 4 * L.mp + (size_t)nmod * L.mp;
    if (fixed + per_col > cap) return false;
    L.nb = std::min<int64_t>(n, (int64_t)((cap - fixed) / per_col));
    size_t o = 0;
    L.off_eA = o;   o += align256(4 * m);
    L.off_eB = o;   o += align256(4 * n);
    L.off_flag = o; o += 256;
    L.off_A8 = o;   o += align256((size_t)nmod * m * L.kp);
    L.off_B8 = o;   o += align256((size_t)nmod * L.nb * L.kp);
    L.off_C32 = o;  o += align256((size_t)4 * L.mp * L.nb);
    L.off_U8 = o;   o += align256((size_t)nmod * L.mp * L.nb);
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
    unsigned char* U8 = (unsigned char*)(ws + L.off_U8);
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
            scale_contig<<<(unsigned)((nvec + 7) / 8), 256, 0, st>>>(X, nvec, k, sv, e, flag);
        else
            scale_strided<<<(unsigned)((nvec + 127) / 128), 128, 0, st>>>(X, nvec, k, sl, e, flag);
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

        for (int t = 0; t < nmod; ++t) {
            nvtxRangePushA("ozd:gemm");
            OZD_CUBLAS(cublasGemmEx_64(h->cublas, CUBLAS_OP_T, CUBLAS_OP_N, m, nb, L.kp,
                                       &one, A8 + t * slabA, CUDA_R_8I, L.kp,
                                       B8 + t * slabB, CUDA_R_8I, L.kp, &zero, C32,
                                       CUDA_R_32I, L.mp, CUBLAS_COMPUTE_32I,
                                       CUBLAS_GEMM_DEFAULT));
            nvtxRangePop();
            h->info.t_gemm_ms += tm.lap();

            nvtxRangePushA("ozd:reduce");
            const int64_t c4 = slabC / 4;
            reduce_mod<<<(unsigned)((c4 + 255) / 256), 256, 0, st>>>(
                (const int4*)C32, (uchar4*)(U8 + t * slabC), c4, h->crt.mod[t]);
            OZD_CUDA(cudaGetLastError());
            nvtxRangePop();
            h->info.t_reduce_ms += tm.lap();
        }

        nvtxRangePushA("ozd:crt");
        crt<<<(unsigned)((slabC + 255) / 256), 256, 0, st>>>(
            U8, slabC, L.mp, m, nb, j0, eA, eB, alpha, beta, C, ldc);
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
