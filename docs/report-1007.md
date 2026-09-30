# HW2 discussion report (10/7): Adaptive FP64 GEMM emulation on INT8 Tensor Cores

Repo: `/work/u1131128/hw2` on nano4. Every number below comes from a file under
`results/<slurm job>/`, and the job ID is given next to each table.

## 1. Topic and method

`ozdgemm` is a DGEMM (`C = αAB + βC`, all four transpose modes, any leading
dimension) that runs on INT8 Tensor Cores using **Ozaki scheme II**, the CRT
variant from arXiv 2504.08009:

1. **Scale.** Each row of A and column of B is scaled by a power of two so that
   `‖A_i‖·‖B_j‖ < 2^(2P)`, then rounded to integers. The rounding is the only
   approximation.
2. **Residues.** Each integer is reduced modulo N pairwise-coprime moduli ≤ 256
   (256, 255, 253, 251, …), which gives N INT8 matrices per operand.
3. **INT8 GEMMs.** One batched cuBLAS INT8×INT8→INT32 GEMM computes all N
   residue products. With these sizes the products are exact.
4. **CRT.** The FP64 result is rebuilt from the N INT32 residues with the
   Chinese remainder theorem. This uses exact 40-bit chunks, an integer
   division by magic numbers and a TwoSum accumulation, and then undoes the
   scaling.

More moduli means a larger P (N = 14 gives P = 54 bits) and therefore higher
accuracy. The error bound is
`|ΔC| ≤ 4·√k·2^−P·|α|·‖A_i‖‖B_j‖ + 4u(|α||A||B| + |βC|)`, and every test checks it.

**Adaptive part (v2):**
- **Moduli per call.** A cheap INT8 GEMM of `|A|·|B|` computes a lower bound
  on each output entry. From it we get `R = max ‖A_i‖‖B_j‖ / (|A||B|)_ij`, and
  N is the smallest count with `P ≥ log2 R + 2 + ½log2 k − log2 tol`. With
  `tol = 0` the target is DGEMM-level accuracy (`k·2^−53`). If N would exceed
  16, the call falls back to native.
- **Backend per call (AUTO).** A linear cost model fitted from benchmarks
  picks emulation only when it is predicted to beat cuBLAS DGEMM.
- **Safety.** NaN/Inf inputs, degenerate shapes, and cases the accuracy
  target can't reach fall back to native DGEMM. The fallback reason is
  reported through `ozd_get_info`, so no failure is silent.

**Environment.**
- GPU: NCHC nano4, 1× H200 (700 W), with a 16-core share of the node's CPU.
- Software: CUDA 13.1, cuBLAS 13.2 (NVHPC 26.3), and netlib HPL 2.3.
- The spec targets an RTX 5090; §6 explains why this matters.

## 2. Three versions

| Version | Commit | What changed |
|---|---|---|
| v0 baseline | `v0-baseline` | Straightforward implementation: separate scale, convert, per-modulus GEMM, reduce, CRT, all in floating point |
| v1 optimized | `7db8259`, `05007ab` | One-pass scaling; tiled convert with packed stores; batched GEMM; fused reduce + CRT; integer CRT with magic-number division |
| v2 adaptive | `680967f` | Adaptive moduli count, AUTO dispatcher, column blocking under a workspace cap |
| final | `f4537d7`, `ef7e39b` | Plus: two-band adaptive bound; one GEMM per modulus for large k (cuBLAS picks a faster tile than for the batched call); refit cost model |

**Speedup over cuBLAS DGEMM (same node, job 466571, median of 10 runs):**

| Shape (m×n×k) | native ms | v0 (14) | final (14) | final (11) | final (8) |
|---|---:|---:|---:|---:|---:|
| 16384³ | 163.4 | 166.9 → **0.98×** | 108.6 → **1.51×** | 88.5 → 1.85× | 71.2 → 2.30× |
| 8192³ | 19.5 | 28.8 → 0.68× | 16.5 → 1.18× | 13.3 → 1.46× | 9.81 → 1.98× |
| 4096³ | 2.26 | 7.11 → 0.32× | 2.60 → 0.87× | 2.11 → 1.07× | 1.59 → 1.43× |
| 32768²×1024 (HPL update) | 39.7 | 97.1 → **0.41×** | 67.4 → **0.59×** | 56.8 → 0.70× | 37.8 → 1.05× |
| 16384²×256 | 2.50 | 19.4 → 0.13× | 11.2 → 0.22× | 10.3 → 0.24× | 6.25 → 0.40× |

(The first draft of this table, job 461398, showed v2 at 1.23× for 16384³;
the per-modulus GEMM change took it to 1.51×.)

- 14 moduli is the count that matches DGEMM accuracy on well-conditioned
  data. 11 moduli is the smallest count that passes HPL (§4).
- Emulation wins only when k is large, roughly k ≳ 8192 with 14 moduli
  (k ≳ 4096 with 11).
- Native was re-timed at the end of each shape (`native_post`). It drifted
  up to 7% slower, so the speedups above are conservative.

## 3. Correctness

The reference is a double-double GEMM on the GPU (`tests/common.cuh`), which
is exact to about 2^−104.

- **Suite** (`tests/test_ozdgemm.cu`, job 461398): 33 cases plus 4
  dispatcher cases, blocking invariance and determinism. **0 failures.**
- **Fixed-moduli cases:** worst error / bound = 0.33. The cases cover all
  transpose modes, odd shapes, padded leading dimensions, 1×1×1, skinny,
  k = 16…16384, zero vectors, rows scaled by 2^300, and φ-distributed inputs
  with exponent range up to 2^±40.
- **NaN/Inf inputs** fall back to native and give the IEEE result.
- **Adaptive sweep** (job 461134, repeated in 462595 with identical results;
  108 cases in `adaptive.csv`):
  - On the 69 emulated cases the guarantee holds; worst error / tol = 0.057.
  - Uniform data chooses 14 moduli at tol = DGEMM level, 11 at 1e-9, 8 at
    1e-6 and 6 at 1e-3.
  - **Level-2 bound** (job 462933): φ=2 cases that fell back now run
    emulated at tol 1e-12 to 1e-3. The sweep has 79 of 108 cases emulated
    (was 69), and the worst error / tol is unchanged (0.057). New tests cover
    level 2 with column blocking.
  - 9 cases ask for `tol = 1e-15` on wide-exponent (φ ≥ 1) data. That is
    tighter than native DGEMM's own error there. The library refuses to emulate
    (fallback reason 4) and returns native's result, whose error is up to 5×
    tol. Neither path can meet such a tolerance; the reason code reports it.
- **Blocking and determinism:** results are bitwise identical with and
  without column blocking, and across repeated runs.

## 4. HPL integration

- **Integration.**
  - netlib HPL-2.3 is built with `-DHPL_CALL_CBLAS`.
  - A link-time shim (`hpl/ozd_shim.cu`) defines `cblas_dgemm`.
  - DGEMM calls with M, N ≥ 1024 are copied to the GPU and run through
    `ozdgemm`; smaller calls stay on the CPU BLAS.
  - A 3-line patch page-locks HPL's matrix so the PCIe copies run at full
    speed.
- **Configuration:** 1 rank, NB = 1024.

**Residual vs. moduli** (N = 20000, job 461194; HPL passes when the scaled
residual is < 16):

| Moduli | native | 14 | 12 | 11 | 10 | 9 | 8 |
|---|---|---|---|---|---|---|---|
| Residual | 2.1e-3 | 2.9e-3 | 0.59 | 9.6 | 165 | 2,497 | 36,101 |
| Result | PASS | PASS | PASS | PASS | FAIL | FAIL | FAIL |

The residual grows by about 16× per modulus removed, matching the 2^−P error
model. **11 moduli is the minimum that passes.**

**Score at N = 60000** (job 461201):

| Config | Time (s) | GFLOPS | vs. native | Residual |
|---|---:|---:|---:|---|
| GPU native DGEMM | 66.51 | 2,165 | 1.000 | 2.3e-3 |
| Emulation, 14 moduli | 68.21 | 2,111 | 0.975 | 3.1e-3 |
| Emulation, 12 moduli | 67.33 | 2,139 | 0.988 | 0.35 |
| Emulation, 11 moduli | 67.12 | 2,146 | 0.991 | 6.3 |
| AUTO, tol = 0 | 66.47 | 2,167 | 1.001 (native on all 57 calls) | 2.3e-3 |

For comparison, CPU-only runs at 497 GFLOPS at N = 20000.

**Where the time goes** (native run, shim timers):

| Part | Time (s) |
|---|---:|
| GPU GEMM | 2.47 (emulation, 14 moduli: 4.15) |
| Host→device copy | 10.86 |
| Device→host copy | 10.34 |
| CPU BLAS (small DGEMMs) | 9.87 |
| Rest of HPL (panel factorization, TRSM, row swaps, MPI) | ~33 |

The GPU GEMM is **3.7% of the run**. Even an infinitely fast GEMM would raise
the score by at most about 4%. netlib HPL keeps the matrix in host memory and
moves it over PCIe on every update, so the result is limited by the host and
PCIe, not by DGEMM.

**Overlapping copies and GEMM** (jobs 461491, 462597; N = 50000 with 12 CPUs,
the new per-GPU limit, so absolute times differ from the N = 60000 runs):
- The shim now copies A and B first, then streams C in 2048-column chunks.
- Copy-in, GEMM and copy-out run on three streams, so both PCIe directions
  and the GPU work overlap.
- For fixed moduli each output column depends only on its own column of B,
  so the results are bitwise identical to the serial path. The residuals
  match to every digit.

| Config | Offloaded calls, serial (s) | Overlapped (s) | GPU GEMM busy (s) |
|---|---:|---:|---:|
| Native | 13.80 | 8.04 | 1.47 |
| Emulation, 14 moduli, first version | 14.77 | 9.73 | 5.30 |
| Emulation, 14 moduli, after the flag fix | — | **8.15** | **2.79** |

HPL time dropped 14% for native (48.8 → 41.8 s, same node), i.e. +16.6% GFLOPS.

The first overlapped version made emulated GEMMs 2.2× slower, and larger
chunks (4096, 8192; job 462480) did not help. The cause was the library's
NaN/Inf check:
- It read an 8-byte flag back with `cudaMemcpyAsync`, which uses the
  device-to-host copy engine.
- That readback queued behind the previous chunk's copy-out (~30 ms), so each
  emulated chunk stalled. Native calls have no readback.

A one-thread kernel now stores the flags into mapped pinned memory instead.
With it, emulated GEMM time matches a run with the check disabled (2.79 vs.
2.78 s), and emulation hides behind the copies almost as well as native
(8.15 vs. 8.04 s). The fix helps any caller that overlaps its own copies with
`ozd_dgemm`.

**Fewer moduli under overlap** (job 463015, N = 50000, same node):

| Config | GPU GEMM busy (s) | Offloaded wall (s) | HPL GFLOPS | Residual |
|---|---:|---:|---:|---|
| Native | 1.46 | 8.38 | 1,954 | 2.1e-3 |
| Emulation, 14 moduli | 2.79 | 8.74 | 2,037 | 2.4e-3 |
| Emulation, 12 moduli | 2.36 | 8.60 | 2,026 | 0.40 |
| Emulation, 11 moduli | 2.26 | 8.74 | 2,031 | 6.4 |

- Fewer moduli shorten the GPU GEMM (2.79 → 2.26 s), but the offloaded
  wall time stays at 8.4–8.7 s. The host-to-device copies (~8 s) set the
  pace.
- The 4% HPL gap in favor of the emulated runs is host-side noise, not a
  gain. Native reached 41.8 s in job 461491 with the same GPU-side time.
- **Conclusion:** once overlapped, HPL on this setup is PCIe-bound. The
  moduli count only changes the residual (all four pass).

## 5. Profiling (Nsight Systems, jobs 461398 and 466571)

**HPL shape 32768²×1024, 14 moduli, per call:**

| Kernel | ms | Share | Note |
|---|---:|---:|---|
| INT8 GEMM (cuBLAS, 14 batched) | 35.5 | 54% | Compute-only bound is 15.6 ms; it also writes 60 GB of INT32 |
| CRT | 28.5 | 43% | Reads 60 GB INT32 and writes 8.6 GB at ~2.4 TB/s, 50% of peak |
| Convert A/B | 1.4 | 2% | |
| Scale | 0.25 | <1% | |
| **Total** | **65.6** | | vs. native DGEMM 38.3 |

**16384³, adaptive tol = 0, per call:**

| Kernel | Before the GEMM change (461398) | After (466571) |
|---|---:|---:|
| INT8 GEMMs (moduli + estimate) | 117.2 | 97.6 |
| Convert | 13.1 | 13.1 |
| CRT | 8.7 | 9.0 |
| Rest of the estimate pass (abs convert + ratio) | 5.6 | 5.5 |
| **Total** | **~146** | **~127** (native 158) |

**Bottleneck.** At small k, the INT32 intermediate is the whole problem:
- N·m·n·4 bytes are written by the GEMM and read back by the CRT.
- The multiply itself is cheap.

At large k the INT8 GEMM dominates, which is the regime where emulation wins.

**Ceiling on this GPU.** Both GEMMs were measured at 16384³ (Nsight, job
466571, after the per-modulus GEMM change):
- The INT8 GEMM reaches ~1,350 TOPS, and native DGEMM reaches ~56 TFLOPS.
- INT8 is therefore only **~24×** faster than FP64 here.

With N moduli, emulation runs N INT8 GEMMs. Even with free conversion and
CRT, its speedup is at most 24/N:

| Moduli | Ceiling | Measured at 16384³ | Share of ceiling |
|---|---:|---:|---:|
| 14 | 1.72× | 1.51× | 88% |
| 11 | 2.19× | 1.85× | 84% |
| 8 | 3.01× | 2.30× | 76% |

Emulation is therefore within 12–24% of the best any INT8 scheme can do on
an H200. Where it loses to native, the reason is that H200 FP64 is strong, not
that the implementation is slow.

## 6. Current problems

1. **HPL cannot show a gain from emulation on this setup.**
   - Serial offload: the GEMM is only ~4% of netlib HPL's time on 1 rank.
   - With overlap: the GEMM hides behind the PCIe copies for both native and
     emulated (§4).
   - Either way, HPL's score does not depend on DGEMM speed here.
   - A GPU-resident HPL would change that, but it is a separate
     multi-thousand-line project. That is the approach of the existing
     GEMMul8-based HPL we used only as a reference.
2. **Native FP64 on the H200 is strong** (67 TFLOPS peak, 1/30 of INT8; see
   the ceiling in §5). Our model, which
   matches the measurements to within 5%, predicts the following on the
   spec's RTX 5090 (FP64 1.6 TFLOPS, INT8 838 TOPS, 1.8 TB/s):
   - 16384³: ~21× over native with 14 moduli.
   - HPL shape: ~10× over native.

   This comes from `scripts/project_gpu.py`, which rescales the fitted
   model to the datasheet peaks. It is a projection, not a measurement.
   Most of our "emulation loses" results are specific to the H200.
3. **The adaptive bound was too pessimistic for wide exponent ranges;
   fixed for φ ≈ 2.**
   - The level-1 `|A||B|` estimate quantizes to 7 bits and rounds some
     entries to 0. For φ ≥ 2 data every tolerance fell back to native.
   - Before building a fix we computed the exact ratio offline (numpy,
     1024³):

     | Data | Exact max log2 R | Level 1 | Two-band (level 2) |
     |---|---:|---|---:|
     | φ=2 | 12.4 | 0.8% zero bounds → fallback | 15 |
     | φ=4 | 28.0 | 69% zero bounds | 0.4% zero bounds |

   - The first draft of this report claimed 14 moduli "would have been
     enough" at `tol = 0`. That was wrong: even the exact R needs more than
     16 moduli there, so falling back is correct.
   - The real loss was φ=2 at looser tolerances. Level 2 (a two-band bound,
     run only when level 1 fails) recovers it: 11/13/15 moduli at
     1e-6/1e-8/1e-10.
   - Level 2 costs four INT8 GEMMs (~29 ms at 16384³). On the H200 it pays
     off only at large k and loose tolerance (1.15× native at 16384³,
     tol 1e-6). φ=4 still falls back.
4. **Small k and skinny shapes are 3–10× slower than native.** AUTO routes
   them to native, so users don't lose time, but emulation has nothing to
   offer there.
5. **Measurement pitfalls we hit.**
   - An A/B of v1 vs. v2 first showed v2 6–8% slower on small shapes (job
     461435). Only the v2 run had timed native DGEMM just before, which heats
     the GPU and slows the next runs by ~5%.
   - Run the same way (job 461490), v1 and v2 match within 0.5%.
     `scripts/abtest.sbatch` now runs both builds identically.
   - Total HPL time also varies by up to ~20% between jobs on shared nodes
     (host-side work). We therefore compare the shim's offload wall time
     within one job.

## 7. Plan for the last week (10/8–10/13)

We first considered a fused INT8 GEMM + mod epilogue (CUTLASS). It would
store residues as uint8 and cut the INT32 traffic 4×. Costed properly at the
INT8 rate we actually achieve (not the peak), it does not pay off:
- The INT8 work alone takes ≥ 28 ms at the HPL shape with 14 moduli.
- The fused path would total ~35–40 ms vs. 38 ms native, i.e. break-even.
  With 11 moduli it would reach ~1.2×.
- In HPL, where the GEMM is 4% of the time, that is < 1% of the score.

It stays optional.

**Done since the first draft:**
- Copy/compute overlap in the HPL shim (§4).
- The mapped-memory flag fix (§4).
- The v1 vs. v2 A/B: no regression (§6).
- The two-band (level-2) adaptive bound (§3, §6).
- HPL with 11–12 moduli under overlap: no change, PCIe-bound (§4).
- One INT8 GEMM per modulus for large k: the stage breakdown showed v1's
  batched GEMM 15% slower than v0's, and Nsight showed cuBLAS choosing a
  smaller tile for batched calls. 16384³ went from 1.23× to 1.51× (A/B job
  466399; the shape rule keeps the batched call where the loop loses).
- HPL shim allocates everything before the timed solve (`late_allocs=0`,
  job 466542).

**Remaining:**
1. **Final report and README.** Add the failure cases, the applicable scope
   (large k, and GPUs with weak FP64) and the profiling evidence.

**Questions for the TA:**
- Is cuBLAS's INT8 GEMM acceptable as a building block? Our own v0 serves as
  the baseline, since no starter code was given.
- Is measuring on the H200 instead of the RTX 5090 acceptable, with the
  projection above?

## AI usage

Claude (Claude Code) was used to:
- plan the work,
- write most of the code, tests and scripts,
- run the Slurm jobs,
- draft this report.

Every number here comes from a committed job log, and every design choice
was reviewed. External code is used only as a reference; see the README.
