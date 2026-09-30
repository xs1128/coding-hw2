# ozdgemm — Adaptive FP64 GEMM emulation on INT8 Tensor Cores

HW2, default topic. Every number cites the Slurm job whose raw output is in
`results/<job>/`. The 10/7 discussion report is `docs/report-1007.md`.

## 1. Setup and baseline

- **Platform.**
  - NCHC nano4, 1× H200 (sm_90, 700 W), CUDA 13.1 / cuBLAS 13.2 (NVHPC 26.3).
  - nano4 has no RTX 5090, so everything was
    measured on the H200. §5 projects the results to an RTX 5090.
- **Starter baseline.**
  - No course starter code was available, so we wrote a straightforward
    Ozaki-II implementation first and froze it as tag `v0-baseline` (also
    built as the `baseline-v0` worktree).
  - Every table compares native FP64, v0, and the final version in the same
    job on the same node.
- **Deliverables.**

  | Item | Where |
  |---|---|
  | Library and API | `include/ozdgemm.h`, `src/ozdgemm.cu`; usage in `README.md` |
  | Correctness tests | `tests/` |
  | Benchmark | `bench/bench.cu` |
  | Slurm and build scripts | `scripts/` |
  | HPL integration | `hpl/`, `scripts/build_hpl.sh` |

## 2. Method

1. **Scale.** Each row of A and column of B gets a power-of-two scale so its
   2-norm is < 2^P, then is truncated to an integer. This is the only
   rounding.
2. **Residues.** The integers are reduced mod N pairwise-coprime moduli
   ≤ 256 and stored as int8.
3. **Multiply.** N exact INT8 GEMMs (INT32 accumulate). When k ≥ 2048 and
   m, n ≥ 1024 these are separate cuBLAS calls; otherwise one batched call.
4. **Reconstruct.** A fused kernel reduces each INT32 result mod m_t, runs the
   CRT with exact 40-bit chunks and TwoSum, and unscales.

The error bound is `|ΔC| ≤ 4√k·2^-P·|α|‖A_i‖‖B_j‖ + 4u(|α||A||B| + |βC|)`.
14 moduli (P = 54) give about DGEMM accuracy.

**Adaptive choices.**
- `ozd_set_accuracy(tol)` bounds R = ‖A_i‖‖B_j‖/(|A||B|)_ij with an INT8 GEMM
  of quantized |A|,|B| (level 1). If that bound fails, a finer two-band pass
  runs (level 2). The library then picks the fewest moduli that guarantee
  `tol`, or falls back to native.
- `OZD_BACKEND_AUTO` picks native or emulated per call from a fitted cost
  model.
- Non-finite inputs, degenerate shapes, and unreachable tolerances fall back
  to native. The reason is reported in `ozd_info_t`, so nothing is hidden.

## 3. Test data, sizes and measurement

**Data.**
- `uniform` U(−0.5, 0.5): HPL's own generator.
- `phi<x>` = (U−0.5)·exp(x·N(0,1)): the Ozaki-paper family. φ controls the
  exponent spread (φ = 4 spans about ±40 binades).
- `rowscale`: rows/columns scaled by 2^±300.
- Corner cases: zero vectors, NaN/Inf, 1×1×1, odd shapes, padded leading
  dimensions, all transpose modes.

**Shapes.**

| Group | Shapes | Why |
|---|---|---|
| Squares | 1k–16k | Throughput |
| HPL trailing updates | m = n = 8k–32k, k = 256–2048 | NB-wide updates, as in HPL |
| Skinny | n = 64/256, k = 16384 | Panel / GEMV-like calls |
| Tall | 65536×1024×1024 | Tall update shape |

**Timing** (`bench/bench.cu`).
- Wall time of the full `ozd_dgemm` call (scaling, conversion, INT8 GEMMs,
  CRT) on device-resident data, with CUDA events.
- Median of 10 calls after 3 warmups. The workspace is reserved before
  timing, so no allocation happens in the timed region.
- Native is `cublasDgemm` under the same conditions, timed first and again
  last (`native_post`) to expose clock drift.
- Per-stage times come from a separate, untimed call with stage events and
  are reported apart from end-to-end times.
- HPL is the only host-to-host end-to-end number.

## 4. Correctness

The reference is a double-double GEMM on the GPU, exact to about 2^-104.
Every entry is checked against the error bound (fixed moduli) or against
`tol·|α||A||B|` (adaptive).

- **Suite** (job 466571): 36 cases plus 4 dispatcher checks, blocking
  invariance and determinism. **0 failures.** The worst error/bound is 0.32.
- **NaN/Inf** fall back to native, with IEEE placement matching.
- **Adaptive sweep** (job 466503): 108 cases (6 generators × 6 tolerances ×
  3 k).
  - The 79 emulated cases all meet `tol`; worst error/tol = 0.057.
  - Uniform data picks 14 moduli at `tol = 0` (DGEMM level), 11 at 1e-9, and
    8 at 1e-6.
- **Blocking and determinism:** column-blocked and repeated calls are
  bitwise identical to unblocked and first calls.

## 5. Performance

**End-to-end speed vs. cuBLAS DGEMM** (job 466571, one node; native ms in
parentheses):

| Shape | v0 (14) | Final (14) | Final (11) | Final (8) |
|---|---:|---:|---:|---:|
| 16384³ (163.4 ms) | 0.98× | **1.51×** | 1.85× | 2.30× |
| 12288³ (65.8) | 0.84× | **1.28×** | 1.49× | 2.19× |
| 8192³ (19.5) | 0.68× | **1.18×** | 1.46× | 1.98× |
| 4096³ (2.26) | 0.32× | 0.87× | 1.07× | 1.43× |
| 32768²×1024, HPL update (39.7) | 0.41× | 0.59× | 0.70× | 1.05× |
| 16384²×256 (2.50) | 0.13× | 0.22× | 0.24× | 0.40× |
| 16384×64×16384 (0.68) | 0.02× | 0.08× | 0.10× | 0.13× |

14 moduli matches DGEMM accuracy on well-conditioned data; 11 is the fewest
that passes HPL (§7). The full grid of 21 shapes is in `results/466571/bench.csv`.

**Ablation, 14 moduli, stage times in ms:**

| Version (job) | Change | 16384³ total | Its stages: scale / convert / GEMM / reduce+CRT | HPL-shape total | Its stages: GEMM / reduce+CRT |
|---|---|---:|---|---:|---|
| v0 (435745) | Separate scale, convert, per-modulus GEMM, reduce, CRT | 170.8 | 12.7 / 39.8 / 100.9 / 17.8 | 98.1 | 34.6 / 63.2 |
| v1 (441228) | One-pass scale, tiled convert with packed stores, batched GEMM, reduce fused into CRT | 135.8 | 1.6 / 13.4 / **115.8** / 9.3 | 71.6 | 35.3 / 32.4 |
| v1 + integer CRT (441237) | Magic-number mod instead of FP | 134.1 | 1.6 / 13.5 / 114.5 / 8.6 | 67.5 | 36.1 / 28.1 |
| Final (466503) | Per-modulus GEMM for large k (A/B 466399) | **110.6** | 1.5 / 12.9 / **86.8** / 8.5 | 67.5 | 36.6 / 28.8 |

v1's batched GEMM regressed the GEMM stage by 15% at 16384³. The per-stage
breakdown exposed it, and Nsight explained it: cuBLAS picks a 64×128 tile
for the batched call and a 128×256 tile for single calls. The loop is 1.20×
faster at 16384³ but 0.70× at 1024³ because of launch overhead, hence the
shape rule.

**Profiling** (Nsight Systems, job 466571):
- **HPL shape, per call:** INT8 GEMM 35.5 ms (54%), CRT 28.5 ms (43%),
  convert 1.4 ms. Both big stages are bound by the 60 GB INT32 intermediate:
  the GEMM writes it and the CRT reads it at ~2.4 TB/s.
- **16384³ adaptive, per call:** INT8 GEMMs 97.6 ms at ~1,350 TOPS
  (including the estimate's), convert 13 ms, CRT 9 ms, the rest of the
  estimate 5.5 ms. That is ~127 ms vs. native 158 ms.

**Ceiling.**
- INT8 runs at ~1,350 TOPS in practice vs. 56 TFLOPS for FP64, only ~24×.
- N moduli therefore cap the speedup at 24/N×.
- We reach 88% of that cap at 14 moduli and 84% at 11.
- Emulation loses where it does because H200 FP64 is strong, not because
  the kernels are slow.
- A fused uint8 epilogue, costed at the achieved INT8 rate, would only break
  even at the HPL shape, so it was not built.

**RTX 5090 projection** (`scripts/project_gpu.py`): the fitted model rescaled
to datasheet peaks (FP64 1.6 TFLOPS, INT8 838 TOPS) predicts ~24× at 16384³
and ~10× at the HPL shape. This is a projection, not a measurement.

**Adaptive and AUTO.**
- The level-1 estimate costs 8% at 16384³ and ~20% at k ≤ 2048.
- Level 2 (4 more INT8 GEMMs) rescues φ=2 data at tol 1e-12 to 1e-3. It
  gains only at large k (1.15× at 16384³, tol 1e-6; job 462933).
- AUTO (model refit on 466503: worst error 16%, 63/63 decisions correct)
  picks the faster path on every tested shape (job 466566).

## 6. Failure cases and applicable scope

| Case | Behavior |
|---|---|
| Small k (≤ 1024 at 14 moduli), skinny, small squares | 0.1–0.9× native; AUTO routes them to native at no extra cost |
| φ ≥ 2 at `tol = 0`, φ = 4 at any tol | Needs > 16 moduli (exact ratio computed offline: log2 R = 12.4 and 28), so falls back with reason 4, correctly |
| `tol` below native DGEMM's own error (1e-15) | Falls back; native's result can still miss `tol`, and the reason code says so |
| k > 131056 | INT32 accumulator limit; falls back |

**Where emulation pays on this GPU:** large k (≳ 4096) with 14 moduli, or
looser tolerances with fewer moduli. On GPUs with weak FP64 it pays almost
everywhere.

## 7. HPL integration

- netlib HPL 2.3 plus a small patch to `HPL_pdtest.c` (page-lock the matrix,
  preallocate before the timed solve).
- A link-time `cblas_dgemm` shim sends updates with M, N ≥ 1024 to
  `ozd_dgemm`.
- 1 rank, NB = 1024.

| Experiment | Result |
|---|---|
| Residual vs. moduli, N=20000 (461194) | 14 → 2.9e-3, 12 → 0.59, **11 → 9.6 (fewest that pass)**, 10 → 165 (fail); ~16× per modulus, as the 2^-P model predicts |
| Serial offload, N=60000 (461201) | Native 2,165 GF, emu:14 2,111, AUTO 2,167. The GEMM is 4% of the time; PCIe copies take 21 s of 66 s |
| Copy/GEMM overlap in C column chunks, N=50000 (461491) | **+16.6% GFLOPS** (48.8 → 41.8 s native); results bitwise identical to serial |
| Emulation under overlap (462597, 463015) | 14/12/11 moduli all hide behind the copies (8.2–8.7 s offloaded vs. 8.0–8.4 s native) |
| No allocation in the timed solve (466542) | `late_allocs=0` in every configuration |

On this setup HPL is bound by PCIe and host-side work, so emulation neither
helps nor hurts it. AUTO correctly keeps native. Beating that needs a
GPU-resident HPL, which is out of scope.

## 8. Debugging lessons (and how to predict a change)

- **Copy-engine stall.** The NaN/Inf flag readback (`cudaMemcpyAsync`)
  queued behind the pipeline's large copies and doubled emulated GEMM time.
  Writing the flag into mapped host memory fixed it (5.05 → 2.79 s).
- **Benchmark order.** A "7% regression" of v2 came from timing native DGEMM
  first, which heats the GPU. A symmetric A/B showed none (461490).
- **HPL timing noise.** Total HPL time varies up to 20% on shared nodes, so
  compare the shim's offload time within one job.
- **Rules of thumb for predictions.**
  - One modulus fewer: ~1/N less GEMM time, 4 bits less accuracy, HPL
    residual ×16.
  - Emulated time ≈ N·(INT8 GEMM) + conversion + CRT, with the INT32 traffic
    dominating at small k.
  - `ozd_get_info` returns the model's `pred_native_ms` / `pred_emu_ms` for
    any shape.

## 9. AI and external code

- **AI.** Claude Code (Claude) was used to plan, write most of the code,
  tests and scripts, run the Slurm jobs, and draft the reports. All numbers
  come from committed job logs; the git history (40 commits) records each
  step.
- **External code.**
  - netlib HPL 2.3 (unmodified tarball plus our patch).
  - cuBLAS INT8 GEMM and DGEMM.
  - GEMMul8 and cuBLAS FP64 emulation are not used. An existing GEMMul8-based
    HPL port was read only as a reference.
- **References.**
  - "Ozaki Scheme II" (arXiv 2504.08009).
  - "Guaranteed DGEMM Accuracy with Automatic Dynamic Precision".
  - "FP8 is All You Need", parts 1 and 2.
  - The GEMMul8 repository, as a reference only.
