# ozdgemm: FP64 GEMM emulation on INT8 Tensor Cores

HW2, default topic. Raw data: `results/<job>/`. Full report: `docs/final-report/report.pdf`.

## Setup
- **Hardware and software:** nano4, 1× H200, CUDA 13.1 / cuBLAS 13.2. nano4
  has no RTX 5090.
- **Baseline:** there is no course starter code, so our own plain version,
  tag `v0-baseline`, serves as the starter. Native FP64, v0 and the final
  version are always compared in the same job on the same node.

## Method
- **Emulation (Ozaki II):**
  - Scale rows and columns by powers of two, then round to integers.
  - Reduce mod N moduli ≤ 256, then run N exact INT8 GEMMs.
  - Rebuild FP64 with the CRT.
  - 14 moduli ≈ DGEMM accuracy.
- **Adaptive:**
  - `ozd_set_accuracy(tol)` bounds the error with an extra INT8 GEMM of
    |A||B|, plus a finer pass if needed, and picks the fewest moduli.
  - `AUTO` picks native or emulated per call from a fitted cost model.
  - NaN/Inf and unreachable tolerances fall back to native, and the reason
    is reported.

## Tests and measurement
- **Data:**
  - uniform (HPL's generator);
  - φ family (Ozaki papers; controls the exponent range);
  - rows scaled by 2^±300;
  - zero vectors, NaN/Inf, odd shapes, all transposes.
- **Shapes:** squares (throughput), HPL updates (k = 256–2048), skinny,
  tall.
- **Timing:**
  - One whole `ozd_dgemm` call on GPU-resident data, median of 10 after 3
    warmups.
  - The workspace is reserved before timing. Stage times come from a
    separate profiled call.

## Correctness
- **Reference:** a double-double GEMM on the GPU.
- **Criterion:** every entry within the proven error bound (fixed moduli),
  or within `tol·|A||B|` (adaptive).
- **Suite:** 36 cases plus dispatcher, blocking and determinism checks. **0
  failures** (job 466571); worst error/bound 0.32.
- **Adaptive sweep:** all 79 emulated cases meet `tol`, worst 0.057 (job
  466503).

## Performance (job 466571, speedup vs. cuBLAS DGEMM)

| Shape | v0 (14 moduli) | Final (14) | Final (11) | Final (8) |
|---|---:|---:|---:|---:|
| 16384³ | 0.98× | **1.51×** | 1.85× | 2.30× |
| 8192³ | 0.68× | **1.18×** | 1.46× | 1.98× |
| 4096³ | 0.32× | 0.87× | 1.07× | 1.43× |
| 32768²×1024 (HPL) | 0.41× | 0.59× | 0.70× | 1.05× |
| 16384×64×16384 | 0.02× | 0.08× | 0.10× | 0.13× |

**Ablation, 16384³, 14 moduli** (ms):

| Version | Time (ms) | What changed |
|---|---:|---|
| v0 | 171 | Baseline |
| v1 | 136 | Faster scale and convert; reduce fused into CRT |
| v1 + integer CRT | 134 | |
| Final | **111** | One GEMM per modulus when k is large |

v1's batched GEMM was 15% slower than v0's. Nsight showed cuBLAS choosing a
smaller tile for the batched call, which the per-modulus loop fixes.

**Profiling and ceiling:**
- At the HPL shape, the INT8 GEMM takes 54% and the CRT 43%. Both are limited
  by the 60 GB INT32 intermediate.
- INT8 is only ~24× faster than FP64 on the H200 (1,350 TOPS vs. 56
  TFLOPS), so N moduli cap the speedup at 24/N×. We reach 88% of that cap.
- An RTX 5090 (weak FP64) is projected at ~10–24× (a model, not a
  measurement).

## HPL (netlib 2.3 + `cblas_dgemm` shim, 1 rank, NB = 1024)
- **Accuracy:** the residual grows ~16× per modulus removed. **11 moduli** is
  the fewest that pass.
- **Speed:** overlapping PCIe copies with the GEMM gives **+16.6%** HPL
  GFLOPS, for native and emulated alike.
- **Emulation:** it then hides behind the copies (8.2 s vs. 8.0 s
  offloaded), so HPL is PCIe-bound and emulation neither helps nor hurts.
  AUTO keeps native.
- **Timed region:** no allocation inside the timed solve (`late_allocs=0`).

## Comparison with NVIDIA's HPL (job 514537)
- **Same H200 node, 12 cores:** NVIDIA HPL (hpc-benchmarks 26.02.01)
  **39.3 TFLOPS** at N = 49152 and 49.6 at N = 131072; ours **2.0** (native
  and emu:14 alike). NVIDIA's DGEMM runs at about the same speed as cuBLAS for us (52.8 vs. 56 TFLOPS);
  its matrix stays on the GPU, so its HPL runs at 94% of DGEMM.
- **NVIDIA's emulation is off on the H200** ("supported only on SM100 or
  newer"): it runs native, bit-identical residual. This matches our finding
  that emulation does not pay at HPL shapes on Hopper.

## Failure cases and scope
- Small k, skinny shapes and small squares run at 0.1–0.9× native; AUTO
  sends them to native.
- φ ≥ 2 data at `tol = 0`, and φ = 4 data, need more than 16 moduli, so they
  fall back. This is correct, as the exact ratio computed offline confirms.
- A `tol` tighter than native's own error (1e-15), or k > 131056, also falls
  back.
- Emulation pays on the H200 for k ≳ 4096–8192, or with looser tolerances.

## AI and external code
- Claude Code wrote most of the code, scripts and reports under my
  direction. All numbers come from committed job logs (40 commits).
- Uses netlib HPL 2.3 (plus a patch) and cuBLAS INT8/FP64 GEMM.
- GEMMul8 and cuBLAS FP64 emulation are not used; a GEMMul8-based HPL was
  read only as a reference.
- References: Ozaki Scheme II (arXiv 2504.08009); Guaranteed DGEMM Accuracy
  with Automatic Dynamic Precision.
