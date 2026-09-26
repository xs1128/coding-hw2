// ozdgemm: FP64 GEMM emulation on INT8 Tensor Cores (Ozaki scheme II, CRT).
//
//   C = alpha * op(A) * op(B) + beta * C      (column-major, device pointers)
//
// All pointers are device pointers. Calls are asynchronous on the handle's
// stream except where noted (the non-finite safety check synchronizes).
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ozd_context* ozd_handle_t;

typedef enum {
    OZD_BACKEND_NATIVE = 0,  // cublasDgemm
    OZD_BACKEND_EMU    = 1,  // INT8 Ozaki-II emulation, always
} ozd_backend_t;

typedef enum { OZD_OP_N = 0, OZD_OP_T = 1 } ozd_op_t;

typedef enum {
    OZD_OK = 0,
    OZD_ERR_ARG = 1,
    OZD_ERR_CUDA = 2,
    OZD_ERR_CUBLAS = 3,
    OZD_ERR_WORKSPACE = 4,
} ozd_status_t;

// What the last ozd_dgemm call actually did (for tests / reporting).
typedef struct {
    int    path;            // 0 = native, 1 = emulated
    int    moduli;          // moduli used (0 if native)
    int    fallback_reason; // 0 none, 1 non-finite input, 2 degenerate, 3 dispatcher
    int    col_blocks;      // number of column blocks the emulation used
    double t_scale_ms;      // stage times, only filled when profiling is on
    double t_convert_ms;
    double t_gemm_ms;
    double t_reduce_ms;     // v0 only (separate INT32 -> uint8 pass); 0 since v1
    double t_crt_ms;
} ozd_info_t;

ozd_status_t ozd_create(ozd_handle_t* h, cudaStream_t stream);
ozd_status_t ozd_destroy(ozd_handle_t h);

ozd_status_t ozd_set_backend(ozd_handle_t h, ozd_backend_t b);
// Number of CRT moduli, 2..16. 14 gives roughly FP64 accuracy.
ozd_status_t ozd_set_moduli(ozd_handle_t h, int moduli);
// Upper bound on workspace bytes (default 4 GiB). The emulation blocks over
// columns of C to stay under it.
ozd_status_t ozd_set_max_workspace(ozd_handle_t h, size_t bytes);
// When on, NaN/Inf inputs are detected and routed to native DGEMM.
// Costs one host sync per call. Default on.
ozd_status_t ozd_set_safety_check(ozd_handle_t h, int on);
// When on, each stage is timed with events (adds syncs; not for benchmarks).
ozd_status_t ozd_set_profiling(ozd_handle_t h, int on);

// Allocate workspace for problems up to (m, n, k) ahead of time so that no
// allocation happens inside timed calls.
ozd_status_t ozd_reserve(ozd_handle_t h, int64_t m, int64_t n, int64_t k);
size_t ozd_workspace_bytes(ozd_handle_t h, int64_t m, int64_t n, int64_t k);

ozd_status_t ozd_dgemm(ozd_handle_t h, ozd_op_t transa, ozd_op_t transb,
                       int64_t m, int64_t n, int64_t k,
                       double alpha, const double* A, int64_t lda,
                       const double* B, int64_t ldb,
                       double beta, double* C, int64_t ldc);

ozd_status_t ozd_get_info(ozd_handle_t h, ozd_info_t* info);
// Scaled operand rows/columns have 2-norm < 2^P for this many moduli. The
// per-entry error bound of the emulation is about 4*sqrt(k)*2^-P*|A_i|*|B_j|.
int ozd_scale_bits(int moduli);
// Implementation version label, used to tag benchmark output.
const char* ozd_version(void);
cublasHandle_t ozd_cublas(ozd_handle_t h);

#ifdef __cplusplus
}
#endif
