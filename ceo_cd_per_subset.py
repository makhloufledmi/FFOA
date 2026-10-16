"""
Generate per-subset Nemenyi critical-difference diagrams (unshifted and
shifted) from ceo_bench_ckpt.json. Highlights the zero-bias collapse
of HHO/GWO/SMA/RUN on the shifted suite, where FFOA emerges as
decisively #1/#2.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from ceo_stats import (
    block_ranks, cd_diagram, cell_median_fes,
    FUNCS, DIMS, ALL_ALGOS,
)

DATA = "ceo_bench_ckpt.json"
OUT = Path("figures")


def build_matrix(data, suite_filter):
    rows, used = [], []
    for s in ["unshifted", "shifted"]:
        if suite_filter and s != suite_filter:
            continue
        for f in FUNCS:
            for d in DIMS:
                vals = [cell_median_fes(data, s, f, d, a) for a in ALL_ALGOS]
                if any(v is None for v in vals):
                    continue
                rows.append(vals)
                used.append((s, f, d))
    return np.asarray(rows, dtype=float), used


def main():
    data = json.load(open(DATA))
    for suite, label in [("unshifted", "Unshifted suite"),
                         ("shifted",   "Shifted suite (zero-bias test)")]:
        X, used = build_matrix(data, suite)
        n = X.shape[0]
        R = block_ranks(X)
        avg = {ALL_ALGOS[k]: float(R[:, k].mean()) for k in range(len(ALL_ALGOS))}
        out_path = OUT / f"cd_{suite}.png"
        cd_diagram(avg, n, out_path,
                   title=f"{label} — {len(ALL_ALGOS)} algorithms, {n} blocks")
        ranking = sorted(avg.items(), key=lambda kv: kv[1])
        print(f"\n=== {suite} ({n} blocks) ===")
        for name, r in ranking:
            print(f"  {name:<10} {r:.3f}")
        print(f"-> {out_path}")


if __name__ == "__main__":
    main()
