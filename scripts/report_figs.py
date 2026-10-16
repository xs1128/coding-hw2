#!/usr/bin/env python3
"""Draw the figures of docs/final-report from committed results.

    python3 scripts/report_figs.py [NV_JOB]      # writes docs/figs/*.svg

NV_JOB is the nvhpl.sbatch job directory (NVIDIA HPL vs. ours); without it the
NVIDIA comparison figure is skipped.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")
OUT = os.path.join(REPO, "docs", "figs")
os.makedirs(OUT, exist_ok=True)

C_NAT, C_V0, C14, C11, C8, C_NV = "#6b7280", "#d97706", "#1d4ed8", "#0891b2", "#16a34a", "#76b900"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "svg.fonttype": "none",
                     "font.family": "DejaVu Sans"})


def rows(path):
    return list(csv.DictReader(open(os.path.join(RES, path))))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name))
    fig.savefig(os.path.join(OUT, name.replace(".svg", ".png")), dpi=200)  # for slides
    plt.close(fig)
    print("wrote", name)


def bench_times():
    t = {}
    for r in rows("466571/bench.csv"):
        key = (int(r["m"]), int(r["n"]), int(r["k"]))
        if r["backend"] == "native":
            t[key, "native"] = float(r["median_ms"])
        elif r["backend"] == "emu_v2":
            t[key, f"emu{r['moduli']}"] = float(r["median_ms"])
        elif r["backend"] == "emu_v0":
            t[key, "v0"] = float(r["median_ms"])
    return t


def fig_squares(t):
    sizes = [1024, 2048, 4096, 8192, 12288, 16384]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for lab, key, c, mk in [("v0 baseline, 14 moduli", "v0", C_V0, "s"),
                            ("final, 14 moduli (≈FP64)", "emu14", C14, "o"),
                            ("final, 11 moduli (HPL min.)", "emu11", C11, "^"),
                            ("final, 8 moduli", "emu8", C8, "v")]:
        ax.plot(sizes, [t[(s, s, s), "native"] / t[(s, s, s), key] for s in sizes],
                marker=mk, color=c, label=lab)
    ax.axhline(1, color=C_NAT, ls="--", lw=1)
    ax.text(1100, 1.05, "cuBLAS DGEMM", color=C_NAT, fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [f"{s}" for s in sizes])
    ax.set_xlabel("m = n = k")
    ax.set_ylabel("speedup vs. native DGEMM")
    ax.set_title("Square GEMM on 1× H200 (job 466571)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    save(fig, "squares.svg")


def fig_hplshape(t):
    ks = [256, 512, 1024, 2048]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, m in zip(axes, [16384, 32768]):
        for lab, key, c, mk in [("v0, 14", "v0", C_V0, "s"), ("final, 14", "emu14", C14, "o"),
                                ("final, 11", "emu11", C11, "^"), ("final, 8", "emu8", C8, "v")]:
            ax.plot(ks, [t[(m, m, k), "native"] / t[(m, m, k), key] for k in ks],
                    marker=mk, color=c, label=lab)
        ax.axhline(1, color=C_NAT, ls="--", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_xlabel("k (HPL: k = NB)")
        ax.set_title(f"m = n = {m}", fontsize=10)
    axes[0].set_ylabel("speedup vs. native")
    axes[1].legend(fontsize=8, frameon=False)
    save(fig, "hplshape.svg")


def fig_ablation():
    labels = ["v0\nbaseline", "v1\nfused kernels", "v1 +\ninteger CRT", "final\nper-modulus GEMM",
              "cuBLAS\nDGEMM"]
    ms = [171, 136, 134, 111, 163.4]
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    bars = ax.bar(labels, ms, color=[C_V0, "#60a5fa", "#3b82f6", C14, C_NAT])
    for b, v in zip(bars, ms):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:g}", ha="center", fontsize=9)
    ax.set_ylabel("ms per call")
    ax.set_title("16384³, 14 moduli: each optimisation step", fontsize=10)
    ax.set_ylim(0, 195)
    ax.tick_params(axis="x", labelsize=8)
    save(fig, "ablation.svg")


def fig_ceiling():
    mod = [14, 11, 8]
    ceil = [24 / n for n in mod]
    meas = [1.51, 1.85, 2.30]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    x = range(len(mod))
    ax.bar([i - 0.18 for i in x], ceil, 0.36, color="#cbd5e1", label="ceiling 24/N")
    ax.bar([i + 0.18 for i in x], meas, 0.36, color=C14, label="measured")
    for i in x:
        ax.text(i + 0.18, meas[i] + 0.05, f"{meas[i] / ceil[i]:.0%}", ha="center", fontsize=8)
    ax.set_xticks(list(x), [f"{n} moduli" for n in mod])
    ax.set_ylabel("speedup vs. native, 16384³")
    ax.legend(fontsize=8, frameon=False)
    save(fig, "ceiling.svg")


def fig_residual():
    r = {x["config"]: float(x["residual"]) for x in rows("461194/summary.csv")}
    labs = ["native", "emu:14", "emu:12", "emu:11", "emu:10", "emu:9", "emu:8"]
    vals = [r[l] for l in labs]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    cols = [C_NAT if l == "native" else (C14 if v < 16 else "#dc2626") for l, v in zip(labs, vals)]
    ax.bar([l.replace("emu:", "") for l in labs], vals, color=cols)
    ax.axhline(16, color="#dc2626", ls="--", lw=1)
    ax.text(6.4, 22, "pass threshold 16", color="#dc2626", ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("moduli")
    ax.set_ylabel("HPL scaled residual")
    ax.set_title("HPL accuracy vs. moduli (N = 20000, job 461194)", fontsize=10)
    save(fig, "residual.svg")


def fig_hpl_breakdown():
    r = {x["config"]: x for x in rows("461201/summary.csv")}
    cfgs = ["native", "emu:14", "emu:11"]
    parts = [("gemm_s", "GPU GEMM", C14), ("h2d_s", "PCIe host→GPU", "#f59e0b"),
             ("d2h_s", "PCIe GPU→host", "#fbbf24"), ("cpu_blas_s", "CPU BLAS (small)", "#a78bfa")]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for i, c in enumerate(cfgs):
        left = 0.0
        for key, lab, col in parts:
            v = float(r[c][key])
            ax.barh(i, v, left=left, color=col, label=lab if i == 0 else None)
            left += v
        rest = float(r[c]["time_s"]) - left
        ax.barh(i, rest, left=left, color="#e5e7eb", label="rest of HPL" if i == 0 else None)
    ax.set_yticks(range(len(cfgs)), ["native", "emu 14", "emu 11"])
    ax.invert_yaxis()
    ax.set_xlabel("seconds (serial offload, N = 60000, job 461201)")
    ax.legend(fontsize=7, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    save(fig, "hplbreakdown.svg")


def fig_nv(job, name="nvhpl.svg", title=None, gpu="H200"):
    path = os.path.join(RES, job, "nv-summary.csv")
    if not os.path.exists(path):
        print("skip nvhpl figure: no", path)
        return
    rs = [r for r in csv.DictReader(open(path)) if r["gflops"]]
    lab = []
    for r in rs:
        if r["impl"] == "nvidia":
            emu = {"refused(sm<100)": ", emulation requested\n(refused on sm_90, ran native)",
                   "ozaki2": ", Ozaki-II emulation", "ozaki1": ", Ozaki-I emulation"}.get(r["nv_emulation"], "")
            lab.append(f"NVIDIA HPL{emu}\nN={int(r['N']) // 1000}k, NB={r['NB']}")
        else:
            lab.append(f"ours netlib+shim\n{r['config']}, N={int(r['N']) // 1000}k")
    vals = [float(r["gflops"]) / 1000 for r in rs]
    cols = [C14 if r["impl"] != "nvidia" else ("#a3e635" if r["nv_emulation"] in ("ozaki1", "ozaki2")
            else C_NV) for r in rs]
    fig, ax = plt.subplots(figsize=(6.8, 0.5 * len(rs) + 0.9))
    bars = ax.barh(range(len(rs)), vals, color=cols)
    for b, v in zip(bars, vals):
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=8)
    ax.set_yticks(range(len(rs)), lab, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"HPL TFLOPS, 1× {gpu}")
    ax.set_title(title or f"NVIDIA HPL vs. our HPL (same node, job {job})", fontsize=10)
    save(fig, name)


if __name__ == "__main__":
    t = bench_times()
    fig_squares(t)
    fig_hplshape(t)
    fig_ablation()
    fig_ceiling()
    fig_residual()
    fig_hpl_breakdown()
    if len(sys.argv) > 1:
        fig_nv(sys.argv[1])
