# ozdgemm: FP64 GEMM emulation on INT8 Tensor Cores

Coding-Basic HW2, default topic (Adaptive FP64 GEMM Emulation + HPL).
Target: NCHC nano4, NVIDIA H200 (sm_90), CUDA 13.1 / cuBLAS 13.2 from NVHPC 26.3.

## Build

```bash
module load x86-nvhpc/26.3   # only needed for MPI/HPL; the library uses absolute paths
make -j                      # builds build/libozdgemm.a, build/test_ozdgemm, build/bench
```

Compiling works on the login node. Run tests and benchmarks inside a Slurm job:

```bash
sbatch scripts/bench.sbatch                                  # tests + all bench presets
BENCH_ARGS="--preset hpl --moduli 14,11" sbatch scripts/bench.sbatch
```

## API

```c
ozd_handle_t h;
ozd_create(&h, stream);
ozd_set_moduli(h, 14);              // 2..16; 14 ~ FP64 accuracy
ozd_reserve(h, m, n, k);            // allocate workspace up front (not in timed code)
ozd_dgemm(h, OZD_OP_N, OZD_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
ozd_get_info(h, &info);             // which path ran, moduli, fallback reason, stage times
ozd_destroy(h);
```

Column-major, device pointers, same semantics as `cublasDgemm`. Backends:
`OZD_BACKEND_NATIVE` (cuBLAS DGEMM) and `OZD_BACKEND_EMU` (INT8 Ozaki-II).

Safety fallbacks to native DGEMM (reported in `info.fallback_reason`):
NaN/Inf in A or B (1), `k == 0`, `alpha == 0`, `k > 131056`, or a problem whose
converted A alone exceeds the workspace cap (2).

## Method (Ozaki scheme II)

1. For each row *i* of op(A) pick a power-of-two exponent `eA[i]` so that the
   scaled row has 2-norm `< 2^P`; same for columns of op(B). `A' = trunc(A·2^eA)`
   is then an integer matrix and `|A'B'|_ij < 2^(2P) <= M/4` by Cauchy–Schwarz.
2. For pairwise coprime moduli `m_t ∈ {256, 255, 253, 251, 247, ...}` compute
   `A' mod m_t` and `B' mod m_t` as int8 (symmetric range).
3. One INT8 GEMM (INT32 accumulate, exact) per modulus.
4. Reduce each INT32 result mod `m_t` to uint8.
5. CRT: `X = Σ c_t S_t mod M` with `S_t = M_t (M_t^-1 mod m_t)`. `S_t` is split
   into 40-bit chunks so every partial sum is exact in FP64; the quotient by `M`
   is found by rounding, and the signed result is summed with TwoSum.
   `C = alpha · X · 2^-(eA+eB) + beta · C`.

`P = floor((floor(log2 M) - 2) / 2)`; for 14 moduli `P = 54`.

## Correctness standard

The only approximation is the truncation `A' = trunc(A·2^eA)`, so each entry of
the result satisfies

    |C - C_exact| <= 4·sqrt(k)·2^-P · |alpha|·||A_i||·||B_j||  +  4u·(|alpha||A||B| + |beta C|)

The test suite (`build/test_ozdgemm`) checks every entry against this bound,
using a double-double reference GEMM. It also reports the componentwise error
`|C - C_ref| / (|A||B|)` next to native DGEMM's. Note that fast-mode scaling
bounds error relative to *row/column norms*, so inputs with a wide exponent
range inside one row lose componentwise accuracy; see the `phi` cases.

Other checks: NaN/Inf inputs fall back to native with identical NaN/Inf
placement; column blocking and repeated calls are bitwise identical.

## Layout

```
include/ozdgemm.h     public API
src/ozdgemm.cu        implementation
tests/                correctness suite and double-double reference
bench/bench.cu        end-to-end benchmark (CSV)
scripts/              Slurm job scripts
results/<jobid>/      raw results (env.txt, test.txt, bench.csv)
third_party/          hpl-2.3.tar.gz from netlib (sha256 32c5c17d...)
```

## AI and external code

Written with an AI coding agent (Claude Code); see the report for details.
GEMMul8 and cuBLAS FP64 emulation are not used in the library. Only cuBLAS
INT8 GEMM (`cublasGemmEx`, `CUBLAS_COMPUTE_32I`) and `cublasDgemm` (native
path) are called.

## Known limitations

- `k` is limited to 131056 on the emulated path (INT32 accumulator); larger `k` falls back.
- The converted copy of op(A) must fit in the workspace cap; column blocks only split B/C.
- The NaN/Inf check synchronizes the stream once per call (can be disabled).
