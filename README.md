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
ozd_set_backend(h, OZD_BACKEND_AUTO); // native or emulated per call (cost model)
ozd_set_accuracy(h, 0.0);           // adaptive moduli, DGEMM-level guarantee
// or: ozd_set_moduli(h, 14);       // fixed 2..16 moduli, 14 ~ FP64 accuracy
ozd_reserve(h, m, n, k);            // allocate workspace up front (not in timed code)
ozd_dgemm(h, OZD_OP_N, OZD_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
ozd_get_info(h, &info);             // path, moduli used, fallback reason, predictions
ozd_destroy(h);
```

Column-major, device pointers, same semantics as `cublasDgemm`. Backends:
`OZD_BACKEND_NATIVE` (cuBLAS DGEMM), `OZD_BACKEND_EMU` (INT8 Ozaki-II) and
`OZD_BACKEND_AUTO` (dispatcher, below).

Fallbacks to native DGEMM (reported in `info.fallback_reason`):
1 NaN/Inf in A or B; 2 `k == 0`, `alpha == 0`, `k > 131056`, or the converted A
alone exceeds the workspace cap; 3 AUTO predicted native to be faster;
4 adaptive tolerance not reachable with 16 moduli.

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

## Adaptive moduli (`ozd_set_accuracy`)

The user states a tolerance `tol`; the call guarantees, for every entry,

    |C - C_exact|_ij <= tol · |alpha| · (|A||B|)_ij   (+ final rounding 4u·(...))

`tol = 0` means native DGEMM's worst-case bound `k·2^-53`. From the bound
above this needs `4·sqrt(k)·2^-P·R_ij <= tol` with
`R_ij = ||A_i||·||B_j|| / (|A||B|)_ij`, so it suffices to bound `max R_ij`:

- `||A_i|| < 2^(norm_i+1)` from the scale pass (already computed).
- A lower bound of `|A||B|` comes from one extra INT8 GEMM of
  `floor(|a|·2^(6-amax_i))` (values 0..127, never above the true scaled |a|).
- A kernel takes the max over entries of `log2` of the resulting upper bound on
  `R_ij`; the host picks the fewest moduli with `P >= that + 2 + log2(sqrt k) - log2 tol`.
  If 16 are not enough, or some entry has a zero lower bound, the call falls
  back to native DGEMM.

Sweep (job 461134, 1024x1024, k in {256, 1024, 4096}): the guarantee held in
every case (worst error/tol 0.06). HPL-like uniform data needs 14 moduli at
`tol = 0`, 11 at 1e-9, 8 at 1e-6. The bound is conservative (errors are
~1-2% of tol, i.e. about 1-2 moduli more than strictly needed), and for
`phi >= 2` data the 7-bit floor quantization zeroes the lower bound for some
entries, so those calls always run natively (safe, but no speedup).

## Dispatcher (`OZD_BACKEND_AUTO`)

A linear cost model (`ozd_cost_model_t`) predicts native and emulated time
from `(m, n, k, moduli)`; the call takes the faster path. When emulation
cannot win even with the fewest admissible moduli the decision is made before
any work, so native calls pay nothing extra. Constants are fitted from
benchmark CSVs by `scripts/fit_model.py` (least squares on relative error)
and can be replaced with `ozd_set_cost_model`. Defaults come from job 441237:
worst prediction error 17%, 63/63 decisions correct on that grid.

## Layout

```
include/ozdgemm.h     public API
src/ozdgemm.cu        implementation
tests/                correctness suite and double-double reference
bench/bench.cu        end-to-end benchmark (CSV)
scripts/              Slurm job scripts, cost-model fit
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
- The NaN/Inf check synchronizes the stream once per call (can be disabled);
  adaptive mode always synchronizes once to choose the moduli count.
- Adaptive mode costs one extra INT8 GEMM (~15-20% at large sizes) and is
  conservative; wide-exponent inputs (`phi >= 2`) always fall back to native.
- The cost model is fitted on one H200; other GPUs need a refit.
