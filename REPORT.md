# ozdgemm — Adaptive FP64 GEMM emulation on INT8 Tensor Cores

HW2, default topic. The detailed write-up is in `docs/report-1007.md`, and the
raw data is in `results/<slurm job>/`.

## Method
- **Ozaki scheme II (CRT).**
  - Rows of A and columns of B are scaled by powers of two and rounded to
    integers, the only approximation.
  - The integers are reduced mod up to 16 coprime moduli ≤ 256.
  - One exact INT8 GEMM per modulus (cuBLAS, INT32 accumulate), then CRT back
    to FP64.
  - 14 moduli ≈ DGEMM accuracy.
- **Adaptive moduli count.** An INT8 GEMM of quantized |A|,|B| bounds the
  error ratio R, and we pick the fewest moduli that meet `tol`. If that
  cheap bound fails (wide exponent range), a finer two-band pass runs;
  otherwise the call falls back to native.
- **AUTO dispatcher.** A fitted cost model chooses native vs. emulated per
  call.
- **Safety.** NaN/Inf, degenerate shapes, and unreachable tolerances fall
  back to native, with the reason reported.
- **HPL.** netlib HPL 2.3 is linked with a `cblas_dgemm` shim that offloads
  large updates to the GPU. Copy-in, GEMM and copy-out are overlapped over
  column chunks.

**Environment:** NCHC nano4, 1× H200, CUDA 13.1 / cuBLAS 13.2 (NVHPC 26.3).

## Results

**Library, speed vs. cuBLAS DGEMM** (job 461398, 14 moduli unless noted):

| Shape | v0 baseline | v2 final | v2, 11 moduli | v2, 8 moduli |
|---|---:|---:|---:|---:|
| 16384³ | 0.98× | **1.23×** | 1.56× | 2.13× |
| 32768²×1024 (HPL update) | 0.41× | 0.59× | 0.70× | 1.05× |
| 4096³ | 0.32× | 0.73× | 0.91× | 1.23× |

Every optimization step sped up every shape. A same-node A/B showed no
v1 → v2 regression (job 461490).

**Why it doesn't beat native more often.** On the H200, INT8 is only ~20×
faster than FP64 in practice (1,112 TOPS vs. 55 TFLOPS). N moduli therefore
cap the speedup at 20/N×, and we reach 85% of that cap. Projected on an
RTX 5090 (weak FP64): ~10–21× faster than native (`scripts/project_gpu.py`,
a model, not a measurement).

**Profile at the HPL shape** (Nsight):
- INT8 GEMM takes 54% and CRT 43%.
- Both are dominated by 60 GB of INT32 intermediates.
- A fused uint8 epilogue would at best break even at 14 moduli, so it was
  not built.

**Correctness:**
- 36 cases plus dispatcher, blocking-invariance and determinism checks:
  0 failures (job 462933).
- Adaptive sweep, 108 cases: every emulated case meets `tol` (worst
  error/tol 0.057), and 79 run emulated.

**HPL** (1 rank):

| | Result |
|---|---|
| Residual vs. moduli (N=20k) | 14 → 2.9e-3, 12 → 0.59, **11 → 9.6 (min passing)**, 10 → 165 (fail) |
| Overlapping copies (N=50k) | **+16.6% GFLOPS**, native and emulated alike; residuals bitwise unchanged |
| Emulation under overlap | 14/12/11 moduli are all hidden behind PCIe (8.6–8.7 s offloaded vs. 8.4 s native) |

HPL on this setup is limited by PCIe and host-side work, not by DGEMM, so
emulation neither helps nor hurts. AUTO correctly keeps native.

## Failure cases and limits
- Emulation loses at small k and on skinny shapes (0.1–0.6×). AUTO routes
  these to native.
- φ=2 data at `tol = 0`, and φ=4 data, need more than 16 moduli, so they
  fall back. The exact ratio, computed offline, confirms this is correct.
- A `tol` tighter than native DGEMM's own accuracy (1e-15) falls back, but
  the result can still miss `tol`; the reason code says so.
- The level-2 bound costs 4 INT8 GEMMs. On the H200 it pays off only at
  large k and loose tolerance (1.15× at 16384³, tol 1e-6).

## Lessons from debugging
- **Copy-engine stall.** The NaN/Inf flag readback (`cudaMemcpyAsync`)
  queued behind large copies on other streams and doubled emulated GEMM
  time in the pipelined shim. Writing the flag into mapped host memory fixed
  it (5.05 → 2.79 s).
- **Benchmark order.** A "6–8% regression" came from timing native DGEMM
  first, which heats the GPU. Compare builds under identical run order.
- **HPL timing noise.** Total HPL time varies ~20% on shared nodes, so
  compare the shim's offload time within one job.

## Reproduce
```bash
make -j && sbatch scripts/bench.sbatch                 # tests + benchmarks
bash scripts/build_hpl.sh && sbatch scripts/hpl.sbatch # HPL (see README)
```

## AI usage
Claude Code (Claude) was used to plan the work, write most of the code,
tests and scripts, run the Slurm jobs, and draft the reports. Every number
comes from a committed job log. GEMMul8 and cuBLAS FP64 emulation are not
used, and the existing GEMMul8-based HPL code was read only as a reference.
