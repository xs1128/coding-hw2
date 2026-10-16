#!/usr/bin/env python3
"""Build nv-summary.csv for an nvhpl.sbatch results directory.

    python3 scripts/nvhpl_summary.py results/<jobid>

One row per run: NVIDIA HPL (nv-*.out) and our netlib HPL + shim (hpl-*.out).
Columns: impl, config, N, NB, HPL time and GFLOPS, scaled residual and
PASSED/FAILED, and whether NVIDIA HPL reported using FP64 emulation.
"""
import glob
import os
import re
import sys

WR = re.compile(r"^W[RC]\S*\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+([\d.]+)\s+([\d.eE+-]+)", re.M)
RES = re.compile(r"_oo/\(eps\*.*=\s*([\d.eE+-]+)\s+\.+\s+(PASSED|FAILED)")


def row(impl, cfg, txt):
    wr, res = WR.search(txt), RES.search(txt)
    f = [impl, cfg]
    f += [wr.group(1), wr.group(2), wr.group(3), wr.group(4)] if wr else ["", "", "", ""]
    f += [res.group(1), res.group(2)] if res else ["", "NONE"]
    if impl == "nvidia":
        if "supported only on SM100" in txt:
            f.append("refused(sm<100)")
        elif "Ozaki-II scheme will be used" in txt:
            f.append("ozaki2")
        elif "GEMM - EMU FP64" in txt:
            f.append("ozaki1")
        else:
            f.append("no")
    else:
        f.append("")
    return ",".join(f)


def main(d):
    rows = ["impl,config,N,NB,time_s,gflops,residual,status,nv_emulation"]
    for log in sorted(glob.glob(os.path.join(d, "nv-*.out")), key=os.path.getmtime):
        if log.endswith("nv-env.txt"):
            continue
        rows.append(row("nvidia", os.path.basename(log)[3:-4], open(log).read()))
    for log in sorted(glob.glob(os.path.join(d, "hpl-*.out")), key=os.path.getmtime):
        rows.append(row("ozd-netlib", os.path.basename(log)[4:-4].replace("-", ":", 1),
                        open(log).read()))
    out = os.path.join(d, "nv-summary.csv")
    open(out, "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main(sys.argv[1])
