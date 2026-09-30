#!/usr/bin/env python3
"""Fit the OZD_BACKEND_AUTO cost model to benchmark CSVs and validate it.

    python3 scripts/fit_model.py results/<job>/bench.csv [...] [--version v1]

Model (same as ozd_cost_model_t in include/ozdgemm.h; times in ms):
    native = o_n + 2mnk / F64
    emu(N) = o_e + (m+n)k(8+N)/BWconv + N*(2mnk/F8 + 4mn/BWout) + (4N+8)mn/BWcrt

Each model is linear in (overheads, 1/rates), so it is fitted by least squares
on relative error (rows divided by the measured time). A coefficient that
comes out negative is dropped and the fit repeated. Only rows of one emulator
version (fixed moduli, emulated path) are used for the emulation fit.
Pure Python, no numpy needed.
"""
import csv
import sys


def lstsq(X, y):
    """Least squares via normal equations (few unknowns, well scaled)."""
    n = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) for j in range(n)] for i in range(n)]
    b = [sum(r[i] * yi for r, yi in zip(X, y)) for i in range(n)]
    for c in range(n):  # Gauss-Jordan with partial pivoting
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        A[c], A[p], b[c], b[p] = A[p], A[c], b[p], b[c]
        for r in range(n):
            if r != c and A[c][c]:
                f = A[r][c] / A[c][c]
                A[r] = [a - f * ac for a, ac in zip(A[r], A[c])]
                b[r] -= f * b[c]
    return [b[i] / A[i][i] if A[i][i] else 0.0 for i in range(n)]


def fit(rows, feats, names):
    """Relative-error least squares with nonnegative coefficients."""
    active = list(range(len(names)))
    while True:
        X = [[feats(r)[i] / r["t"] for i in active] for r in rows]
        y = [1.0] * len(rows)
        coef = lstsq(X, y)
        neg = [a for a, c in zip(active, coef) if c < 0]
        if not neg:
            full = [0.0] * len(names)
            for a, c in zip(active, coef):
                full[a] = c
            return full
        active.remove(neg[0])


def native_feats(r):
    return [1.0, 2 * r["m"] * r["n"] * r["k"]]


def emu_feats(r):
    m, n, k, N = r["m"], r["n"], r["k"], r["N"]
    return [1.0, (m + n) * k * (8 + N), N * 2 * m * n * k, N * 4 * m * n, (4 * N + 8) * m * n]


def main():
    args = sys.argv[1:]
    version = "v1"
    if "--version" in args:
        i = args.index("--version")
        version = args[i + 1]
        del args[i:i + 2]
    nat, emu = [], []
    for path in args:
        with open(path) as f:
            for d in csv.DictReader(f):
                r = {"m": float(d["m"]), "n": float(d["n"]), "k": float(d["k"]),
                     "t": float(d["median_ms"]), "label": d["backend"],
                     "N": int(d["moduli"]), "shape": f'{d["m"]}x{d["n"]}x{d["k"]}'}
                if d["backend"] == "native":
                    nat.append(r)
                elif d["backend"] == "emu_" + version and r["N"] > 0 and d.get("path", "1") == "1":
                    emu.append(r)
    if not nat or not emu:
        sys.exit("need native rows and emu_%s rows" % version)

    cn = fit(nat, native_feats, ["o_n", "1/F64"])
    ce = fit(emu, emu_feats, ["o_e", "1/conv", "1/F8", "1/out", "1/crt"])
    inv = lambda c, s: s / c if c > 0 else float("inf")
    model = {
        "native_overhead_ms": cn[0],
        "native_tflops": inv(cn[1], 1e-9),       # ms -> TFLOP/s: 2mnk flop / (TF*1e9) ms
        "emu_overhead_ms": ce[0],
        "int8_tops": inv(ce[2], 1e-9),
        "convert_gbs": inv(ce[1], 1e-6),         # bytes / (GB/s * 1e6) ms
        "gemm_out_gbs": inv(ce[3], 1e-6),
        "crt_gbs": inv(ce[4], 1e-6),
    }
    print("fitted model:")
    for k, v in model.items():
        print(f"  {k:20s} {v:12.4f}")
    print("C initializer:")
    print("  {" + ", ".join(f"{v:.4g}" if v != float("inf") else "1e30" for v in model.values()) + "}")

    def pred(r, feats, c):
        return sum(f * x for f, x in zip(feats(r), c))

    print("\nvalidation (pred vs measured, ms):")
    print(f"  {'shape':22s} {'cfg':8s} {'meas':>9s} {'pred':>9s} {'err':>7s}")
    worst = 0.0
    for r in nat + emu:
        c, feats = (cn, native_feats) if r["label"] == "native" else (ce, emu_feats)
        p = pred(r, feats, c)
        e = p / r["t"] - 1
        worst = max(worst, abs(e))
        cfg = "native" if r["label"] == "native" else f"N={r['N']}"
        print(f"  {r['shape']:22s} {cfg:8s} {r['t']:9.3f} {p:9.3f} {100 * e:6.1f}%")
    print(f"worst relative error {100 * worst:.1f}%")

    # Dispatch quality: for each (shape, N), does the model pick the faster path?
    tn = {r["shape"]: r["t"] for r in nat}
    right = total = 0
    lost = 0.0
    for r in emu:
        if r["shape"] not in tn:
            continue
        pick_emu = pred(r, emu_feats, ce) < pred({**r}, native_feats, cn)
        best_emu = r["t"] < tn[r["shape"]]
        total += 1
        right += pick_emu == best_emu
        chosen = r["t"] if pick_emu else tn[r["shape"]]
        lost = max(lost, chosen / min(r["t"], tn[r["shape"]]) - 1)
    print(f"dispatch decisions correct: {right}/{total}; worst slowdown vs. best choice {100 * lost:.1f}%")


if __name__ == "__main__":
    main()
