#!/usr/bin/env python3
"""Rebuild summary.csv for an HPL results directory from its hpl-*.out logs.

    python3 scripts/hpl_summary.py results/<jobid>

Columns: config, N, NB, HPL time and GFLOPS, scaled residual and PASSED/FAILED,
then the shim's statistics (GPU calls, emulated calls, moduli range, total
offloaded GFLOP, and seconds in host->device copies, GPU GEMM, device->host
copies and CPU BLAS DGEMM, which overlap when the shim pipelines; shim_wall_s
is the wall time of the offloaded calls; late_allocs counts device buffer
allocations made after setup, i.e. inside HPL's timed solve, and should be 0).
"""
import glob
import os
import re
import sys

WR = re.compile(r"^WR\S+\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+([\d.]+)\s+([\d.eE+-]+)", re.M)
RES = re.compile(r"_oo/\(eps\*.*=\s*([\d.eE+-]+)\s+\.+\s+(PASSED|FAILED)")
CALLS = re.compile(r"calls cpu=(\d+) gpu=(\d+) \(emulated (\d+), moduli (\d+)\.\.(\d+)\)")
FLOP = re.compile(r"gflop cpu=([\d.]+) gpu=([\d.]+) \| time s: cpu_blas=([\d.]+) h2d=([\d.]+) "
                  r"gemm=([\d.]+) d2h=([\d.]+)")
WALL = re.compile(r"d2h=[\d.]+ wall=([\d.]+)")
LATE = re.compile(r"late_allocs=(\d+)")


def main(d):
    rows = ["config,N,NB,time_s,gflops,residual,status,gpu_calls,emulated,moduli,"
            "gpu_gflop,h2d_s,gemm_s,d2h_s,cpu_blas_s,shim_wall_s,late_allocs"]
    for log in sorted(glob.glob(os.path.join(d, "hpl-*.out")), key=os.path.getmtime):
        cfg = os.path.basename(log)[4:-4].replace("-", ":", 1)
        txt = open(log).read()
        wr, res = WR.search(txt), RES.search(txt)
        calls, flop = CALLS.search(txt), FLOP.search(txt)
        f = [cfg]
        f += [wr.group(1), wr.group(2), wr.group(3), wr.group(4)] if wr else ["", "", "", ""]
        f += [res.group(1), res.group(2)] if res else ["", "NONE"]
        if calls:
            mod = calls.group(4) if calls.group(4) == calls.group(5) else f"{calls.group(4)}-{calls.group(5)}"
            f += [calls.group(2), calls.group(3), mod if calls.group(3) != "0" else "0"]
        else:
            f += ["", "", ""]
        f += [flop.group(2), flop.group(4), flop.group(5), flop.group(6), flop.group(3)] if flop else [""] * 5
        wall = WALL.search(txt)
        f += [wall.group(1) if wall else ""]
        late = LATE.search(txt)
        f += [late.group(1) if late else ""]
        rows.append(",".join(f))
    out = os.path.join(d, "summary.csv")
    open(out, "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main(sys.argv[1])
