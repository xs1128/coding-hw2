#!/usr/bin/env python3
"""Project emulated vs. native DGEMM time on another GPU from the H200 cost model.

    python3 scripts/project_gpu.py

This is a paper estimate, not a measurement. Each term of the fitted model
(src/ozdgemm.cu, kDefaultModel, job 441237) is scaled by the ratio of the
target's datasheet peak to the H200's: INT8 tensor throughput for the GEMM
term, DRAM bandwidth for the convert/CRT/output terms, FP64 for native.
Launch overheads are kept as measured.
"""

H200 = dict(int8=1979, bw=4.8, fp64=67.0)        # dense TOPS, TB/s, TFLOPS
TARGETS = {
    "H200 (measured model)": H200,
    "RTX 5090": dict(int8=838, bw=1.792, fp64=1.637),
}
MODEL = dict(native_overhead_ms=0.01265, native_tflops=56.66, emu_overhead_ms=0.05899,
             int8_tops=1182.0, convert_gbs=949.4, gemm_out_gbs=1637.0)
SHAPES = [(16384, 16384, 16384), (32768, 32768, 1024), (8192, 8192, 8192), (4096, 4096, 4096)]
MODULI = [14, 11, 8]


def scaled(t):
    r_i, r_b, r_f = t["int8"] / H200["int8"], t["bw"] / H200["bw"], t["fp64"] / H200["fp64"]
    return dict(MODEL, native_tflops=MODEL["native_tflops"] * r_f, int8_tops=MODEL["int8_tops"] * r_i,
                convert_gbs=MODEL["convert_gbs"] * r_b, gemm_out_gbs=MODEL["gemm_out_gbs"] * r_b)


def native(c, m, n, k):
    return c["native_overhead_ms"] + 2 * m * n * k / (c["native_tflops"] * 1e9)


def emu(c, m, n, k, N):
    # The fit drove the separate CRT term to zero (crt_gbs = 1e30): CRT cost is
    # absorbed into the per-modulus output term, which scales with bandwidth.
    kp = (k + 15) // 16 * 16
    conv = (m + n) * kp * (8 + N) / (c["convert_gbs"] * 1e6)
    gemm1 = 2 * m * n * kp / (c["int8_tops"] * 1e9) + 4 * m * n / (c["gemm_out_gbs"] * 1e6)
    return c["emu_overhead_ms"] + conv + N * gemm1


print("gpu,m,n,k,native_ms," + ",".join(f"emu{N}_ms,speedup{N}" for N in MODULI))
for name, t in TARGETS.items():
    c = scaled(t)
    for m, n, k in SHAPES:
        tn = native(c, m, n, k)
        cols = []
        for N in MODULI:
            te = emu(c, m, n, k, N)
            cols += [f"{te:.2f}", f"{tn / te:.2f}"]
        print(f"{name},{m},{n},{k},{tn:.2f}," + ",".join(cols))
