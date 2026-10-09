# Video script (≤ 10 min)

Narration outline for a planned slide deck "ozdgemm HW2 Talk".
15 sections; the script is intended to fit within 10 minutes, including pauses and chart explanations.
Audience: a teammate who has not seen the topic.

## 1. ozdgemm
*~25 s*

Hi. This is a walkthrough of our HW2 project for teammates who haven't seen the topic. In ten minutes: the problem, how INT8 hardware can compute exact FP64 products, how we guarantee accuracy, what we measured on an H200, how it behaves inside HPL, and how it compares with NVIDIA's own HPL. Every number comes from a Slurm job log in the repo.

## 2. The problem: GPUs are getting fast at INT8, not at FP64
*~36 s*

GPUs keep adding low-precision throughput while FP64 stays flat. On our H200, INT8 is about 24 times faster than FP64 in practice; on the RTX 5090 in the spec it is about 500 times on paper. If we can rebuild an exact FP64 matrix product from INT8 products, scientific codes keep FP64 accuracy at INT8 speed. Keep three questions in mind: how many INT8 products we need, how to guarantee the accuracy, and when it is actually faster. The H200 is a hard case, because its FP64 is strong.

## 3. The idea: store a big integer as small remainders
*~38 s*

The core trick. Store 30 only as its remainders mod 7 and mod 11: 2 and 8. Since 7 times 11 is 77, every integer from minus 38 to 38 has its own pair, so 2, 8 can only mean 30. That is the Chinese remainder theorem. Ozaki scheme two applies it to matrices: scale to integers, take remainders up to 256 so they fit in INT8, multiply them exactly on tensor cores, and rebuild with the CRT. 14 moduli give about 54 bits, roughly FP64. Each extra modulus buys bits and costs one INT8 GEMM.

## 4. One ozd_dgemm call, step by step
*~36 s*

One library call. Scale rows and columns by powers of two, so scaling is exact. Convert to N INT8 copies. Run N INT8 GEMMs in cuBLAS, exact in INT32. One fused kernel reduces and runs the CRT, with 40-bit pieces so every FP64 sum is exact. Then undo the scaling. Around this, the adaptive part picks how many moduli the requested tolerance needs, and AUTO uses a fitted cost model to pick native DGEMM when emulation would lose. NaN or impossible tolerances fall back to native, always with a reason code.

## 5. Accuracy is proven, then the moduli count is chosen
*~38 s*

The only approximation is rounding to integers, which gives this bound; P is the bits we keep. The user gives a tolerance relative to |A| times |B|, the same yardstick as DGEMM; zero means match native. Level one bounds |A||B| from below with one extra INT8 GEMM and picks the fewest moduli. If exponents are too spread out, level two uses four more GEMMs. If 16 moduli still aren't enough, we return native. On HPL-like data it picks 14 moduli at DGEMM accuracy, 11 at 1e-9, 8 at 1e-6, and every case met its tolerance.

## 6. Correctness: every entry, against an exact reference
*~30 s*

We check every entry, not an average. The reference is a double-double GEMM, accurate to 2 to the minus 104, and each entry must fall inside the proven bound. 36 cases cover HPL's uniform data, the phi family with spread-out exponents, rows scaled by 2 to the 300, zero vectors, NaN and Inf, every transpose and odd shapes. Zero failures; the worst entry used 32 percent of its bound. Blocked and repeated runs are bitwise identical.

## 7. GEMM speed: wins at large k, loses at small k
*~34 s*

Speedup over cuBLAS DGEMM against matrix size. The dotted line with square markers is our first plain version; the solid line with circles is the final one at 14 moduli, which is FP64 accuracy. It wins from about 8192 cubed: 1.18 times, and 1.51 at 16384 cubed. With 11 or 8 moduli it reaches 1.85 and 2.3. Fused kernels, an integer CRT and one cuBLAS call per modulus took 16384 cubed from 171 to 108 milliseconds. But at HPL's update shape, k equals 1024, it is only 0.59 times. Small k is where we lose.

## 8. Why: the H200 caps emulation at 24/N
*~32 s*

Why? At large k, N moduli means N INT8 GEMMs, and INT8 is only 24 times FP64 here, so the best possible speedup is 24 over N: 1.7 for 14 moduli. We reach 88 percent of that. At small k the multiply is cheap, but each modulus writes a full INT32 matrix: 60 gigabytes at the HPL shape, written and read back, so both kernels are memory-bound. A fused kernel would only break even with native, so we dropped it.

## 9. HPL accuracy: 11 moduli is the minimum that passes
*~25 s*

Now HPL. We kept netlib HPL 2.3 unmodified and added a link-time shim for cblas_dgemm that sends big updates to the GPU through our library. HPL passes if its scaled residual stays below 16. Each modulus removed costs about 4 bits, so the residual grows about 16 times per modulus, exactly as the bound predicts. 11 moduli is the fewest that passes.

## 10. HPL speed: our score does not move, ~2.0 TFLOPS
*~31 s*

Correct, but not faster, and this chart shows why. At N equals 60000 the GPU GEMM is 2.5 of 66 seconds. netlib HPL keeps the matrix in host memory, so every update crosses PCIe. Native and emulated all land near 2 TFLOPS. Overlapping copies with the GEMM gave 16.6 percent for both, and after fixing a flag read-back that stalled the copies, the emulated GEMM hides behind PCIe entirely. So this HPL can't show a GEMM speedup.

## 11. vs. NVIDIA's HPL on the same H200: 39.3 vs 2.0 TFLOPS
*~36 s*

The TA asked us to compare with NVIDIA's HPL, so we ran NVIDIA's container on the same node, same job, same 12 cores. It gets 39.3 TFLOPS at N about 50 thousand and 49.6 at 131 thousand; ours gets 2.0. Not because of DGEMM: its DGEMM test runs at 52.8, like our cuBLAS. It keeps the matrix on the GPU and runs at 94 percent of DGEMM speed. And its built-in Ozaki-II emulation refuses to run on the H200, "SM100 or newer" only, which matches our finding for Hopper.

## 12. RTX 5090: a projection to test, not a measured result
*~30 s*

The other hardware type for this project is the RTX 5090. We do not use a measured RTX 5090 benchmark in this report. Instead, the cost-model script rescales H200 measurements using assumed RTX 5090 compute and memory rates. It estimates about 21 times native DGEMM at 16384 cubed and about 10 times at the HPL update shape. These are model predictions, not experimental speedups. A real RTX 5090 run is needed before claiming those gains.

## 13. Where it fails, and where it pays
*~33 s*

To be fair about scope: small-k, skinny and small problems run at 0.1 to 0.9 times native, and AUTO sends them to native. Very spread-out exponents at DGEMM accuracy need more than 16 moduli, which we confirmed with an exact offline check. Over-tight tolerances and huge k also fall back, with a reason code. It pays on the H200 for large k or looser tolerances, and is projected at 10 to 21 times on an RTX 5090, a model, not a measurement.

## 14. How we worked with an AI agent (Claude Code)
*~32 s*

Claude Code wrote most of the code, tests and job scripts, ran the Slurm jobs and drafted the reports; the student set the goals and rules and reviewed every result. Two rules made it trustworthy: every number cites a committed job log, and tests come first. They caught real mistakes: an over-claim about wide-exponent data, a fake regression from GPU heating, and a flag read-back that stalled the pipeline. The agent is fast; trust comes from logs, tests and same-job baselines.

## 15. Takeaways
*~32 s*

Three takeaways. One: INT8 tensor cores can compute FP64 products with a proven per-entry bound; 14 moduli match FP64. Two: on the H200 that gives up to 1.5 times for large k, and the adaptive dispatcher keeps it safe elsewhere. Three: HPL gains only if the HPL is GPU-resident and DGEMM-bound, with a large block size, on a GPU where INT8 far outruns FP64. The RTX 5090 estimates are projections that still need measurement. Code, report and logs are in the repo. Thanks.
