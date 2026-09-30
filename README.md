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
4 adaptive tolerance not reachable with 16 moduli. A fallback returns native
DGEMM's result, so with reason 4 and a `tol` tighter than native DGEMM itself
achieves (e.g. `1e-15` on wide-exponent data) the result can still miss `tol`;
the reason code tells the caller.

## Method (Ozaki scheme II)

1. For each row *i* of op(A) pick a power-of-two exponent `eA[i]` so that the
   scaled row has 2-norm `< 2^P`; same for columns of op(B). `A' = trunc(A·2^eA)`
   is then an integer matrix and `|A'B'|_ij < 2^(2P) <= M/4` by Cauchy–Schwarz.
2. For pairwise coprime moduli `m_t ∈ {256, 255, 253, 251, 247, ...}` compute
   `A' mod m_t` and `B' mod m_t` as int8 (symmetric range).
3. One INT8 GEMM (INT32 accumulate, exact) per modulus.
4. Reduce each INT32 result mod `m_t` (integer magic-number division, fused
   into the CRT kernel).
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
- **Level 2.** For wide exponent ranges the 7-bit floor rounds some lower
  bounds to 0 (or leaves them too loose), and level 1 certifies no count. Then
  `s = |a|·2^(6-amax)` is split into `h = floor(s)` and `l = floor((s-h)·128)`
  (both exact, `s >= h + l/128`), and four INT8 GEMMs give
  `|A||B| >= hh + (hl+lh)/128 + ll/16384`, summed in 64 bits. Level 2 runs
  only when level 1 fails; AUTO skips it when native is predicted faster
  even with the fewest moduli. `info.estimate_level` says which level ran.
- If 16 moduli are still not enough, or some entry still has a zero lower
  bound, the call falls back to native DGEMM (reason 4).

Sweep (jobs 461134, 462933; 1024x1024, k in {256, 1024, 4096}): the
guarantee held in every emulated case (worst error/tol 0.057). HPL-like
uniform data needs 14 moduli at `tol = 0`, 11 at 1e-9, 8 at 1e-6, all from
level 1. The bound is conservative (errors are ~1-2% of tol, about 1-2
moduli more than strictly needed). `phi = 2` data now runs emulated through
level 2 at tol 1e-12 to 1e-3 (16 down to 9 moduli); at `tol = 0` it still
falls back, correctly: even the exact ratio (max log2 R = 12.4, computed
offline) would need more than 16 moduli. `phi = 4` data falls back at every
tol (level 2 still has zero bounds).

## Dispatcher (`OZD_BACKEND_AUTO`)

A linear cost model (`ozd_cost_model_t`) predicts native and emulated time
from `(m, n, k, moduli)`; the call takes the faster path. When emulation
cannot win even with the fewest admissible moduli the decision is made before
any work, so native calls pay nothing extra. Constants are fitted from
benchmark CSVs by `scripts/fit_model.py` (least squares on relative error)
and can be replaced with `ozd_set_cost_model`. Defaults come from job 441237:
worst prediction error 17%, 63/63 decisions correct on that grid.

## HPL integration

netlib HPL 2.3 (tarball in `third_party/`, unmodified except a 3-line patch)
is built with `-DHPL_CALL_CBLAS`, and a link-time shim (`hpl/ozd_shim.cu`)
defines `cblas_dgemm`. Updates with `M, N >= 1024` and `K >= 128` go to the GPU
through `ozd_dgemm`; smaller calls stay on the CPU BLAS (OpenBLAS). HPL keeps
its matrix in host memory, so every offloaded update copies A, B and C over
PCIe. The patch (`hpl/hpl-ozd.patch`) page-locks HPL's matrix so those copies
run at full PCIe speed.

```bash
bash scripts/build_hpl.sh          # verifies the tarball, patches, builds build/xhpl
N=50000 CONFIGS="native emu:14 autotol:0" sbatch scripts/hpl.sbatch
python3 scripts/hpl_summary.py results/<jobid>   # (re)builds summary.csv
```

`CONFIGS` entries are `cpu | native | emu:<moduli> | auto:<moduli> |
tol:<tol> | autotol:<tol>`, optionally followed by `,VAR=value` shim settings.
Shim environment (the `HPL_OZD_` prefix is implied in `CONFIGS`):

| Variable | Default | Meaning |
|---|---|---|
| `HPL_OZD_MODE` | `emu` | `cpu`, `native`, `emu` or `auto` |
| `HPL_OZD_MODULI` / `HPL_OZD_TOL` | 14 / unset | fixed moduli, or adaptive tolerance |
| `HPL_OZD_MIN` | 1024 | offload when `M, N >= MIN` and `K >= MIN/8` |
| `HPL_OZD_PIPE` | 1 | stream C in column chunks, overlapping copy-in, GEMM and copy-out |
| `HPL_OZD_CHUNK` | 2048 | chunk width in columns |
| `HPL_OZD_REG` | 1 | page-lock HPL's matrix |
| `HPL_OZD_SAFETY` | 1 | 0 skips the NaN/Inf check (diagnostic) |

At exit the shim prints call counts, moduli used, fallbacks and the time in
host-to-device copies, GPU GEMM, device-to-host copies and CPU BLAS, plus the
wall time of the offloaded calls (with pipelining the first three overlap).

Results (1 rank, NB = 1024):

- **Accuracy** (job 461194, N = 20000): the scaled residual grows ~16x per
  modulus removed; 11 moduli is the smallest count that passes (9.6 < 16).
- **Serial offload** (job 461201, N = 60000, 16 CPUs): native 2,165 GFLOPS,
  emu:14 2,111, emu:11 2,146, AUTO 2,167 (native on every call). GPU GEMM is
  2.5 s of 66.5 s; the PCIe copies (21 s) and host work dominate.
- **Pipelined offload** (jobs 461491, 462597, N = 50000, 12 CPUs): HPL time
  drops 14% (+16.6% GFLOPS) for native; the offloaded calls take 8.0 s (native) and 8.2 s
  (emu:14) instead of 13.8 / 14.8 s, i.e. the emulated GEMM hides behind the
  copies. Residuals are identical to the serial path. With 12 or 11 moduli
  (job 463015) the GPU GEMM shrinks (2.79 -> 2.26 s) but the offloaded time
  does not (8.6-8.7 s): the host-to-device copies set the pace.
- Total HPL time varies by up to ~20% between runs on shared nodes (host-side
  work), so compare `shim_wall_s` in `summary.csv` within one job.

## Layout

```
include/ozdgemm.h     public API
src/ozdgemm.cu        implementation
tests/                correctness suite and double-double reference
bench/bench.cu        end-to-end benchmark (CSV)
hpl/                  cblas_dgemm shim, HPL patch and Make.ozd template
scripts/              Slurm job scripts, HPL build, cost-model fit, projections
results/<jobid>/      raw results (env.txt, test.txt, bench.csv, HPL logs)
docs/                 reports
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
  adaptive mode always synchronizes once to choose the moduli count. The flags
  come back through mapped host memory, not a copy, so this sync does not
  wait behind the caller's device-to-host copies on other streams.
- Adaptive mode costs one extra INT8 GEMM pass (8% at 16384^3, ~20% at
  k <= 2048) and is conservative. The level-2 pass adds four more (~29 ms at
  16384^3, ~35 ms at 32768^2x1024), so on an H200 wide-exponent data gains
  only at large k and loose tol (phi=2, 16384^3, tol 1e-6: 1.15x native);
  `phi >= 4` data always falls back.
- On an H200 native FP64 is strong (INT8 is only ~20x faster in practice), so
  14-moduli emulation wins only for large k (about k >= 8192); the ceiling is
  20/N x for N moduli. See `docs/report-1007.md`.
- Running HPL through the shim cannot show a large gain on one H200: netlib
  HPL keeps the matrix on the host, so PCIe copies and host work dominate.
- The cost model is fitted on one H200; other GPUs need a refit.
