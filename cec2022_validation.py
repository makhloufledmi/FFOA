"""Compact CEC-2022 validation for the canonical FFOA.

Runs CEC-2022 D=10 functions with FE-counted runners. The suite includes
shifted/rotated basic functions, hybrid functions, and composition functions.
Results are checkpointed after each (function, algorithm, seed) run.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from collections import defaultdict

import opfunu

from ceo_comparators import make_runner


DIM = int(os.environ.get("CEC_DIM", "10"))
RUNS = int(os.environ.get("CEC_RUNS", "10"))
POP_SIZE = int(os.environ.get("CEC_POP_SIZE", "30"))
MAX_FES = int(os.environ.get("CEC_MAX_FES", str(DIM * 10_000)))
ERR_TOL = 1e-8
ALGOS = [a.strip() for a in os.environ.get(
    "CEC_ALGOS", "FFOA,LSHADE,SADE,DE,CEO"
).split(",") if a.strip()]
TAG = os.environ.get("CEC_TAG", "validation")
CACHE_FILE = f"cec2022_{TAG}_results.json"
SUMMARY_FILE = f"cec2022_{TAG}_summary.json"

FUNCTIONS = {
    1: "rotated/basic",
    2: "rotated/basic",
    3: "rotated/basic",
    4: "rotated/basic",
    5: "rotated/basic",
    6: "hybrid",
    7: "hybrid",
    8: "hybrid",
    9: "composition",
    10: "composition",
    11: "composition",
    12: "composition",
}


def load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def key(fid: int, algo: str, seed: int) -> str:
    return f"F{fid}|{algo}|seed{seed}"


def rank_values(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda kv: kv[1])
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and math.isclose(ordered[j][1], ordered[i][1], rel_tol=0, abs_tol=1e-12):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for algo, _ in ordered[i:j]:
            ranks[algo] = avg_rank
        i = j
    return ranks


def scored_error(record: dict) -> float:
    error = max(0.0, float(record["error"]))
    return 0.0 if record.get("fes_to_target") is not None or error <= ERR_TOL else error


def summarize(cache: dict) -> dict:
    cells = {}
    rank_lists = defaultdict(list)
    category_rank_lists = defaultdict(lambda: defaultdict(list))
    wins = defaultdict(int)
    hits = defaultdict(int)

    for fid, category in FUNCTIONS.items():
        cell_stats = {}
        for algo in ALGOS:
            vals = [
                scored_error(cache[key(fid, algo, seed)])
                for seed in range(RUNS)
            ]
            fes_hits = [cache[key(fid, algo, seed)]["fes_to_target"] for seed in range(RUNS)]
            cell_stats[algo] = {
                "mean_error": statistics.fmean(vals),
                "median_error": statistics.median(vals),
                "std_error": statistics.pstdev(vals),
                "best_error": min(vals),
                "worst_error": max(vals),
                "hits": sum(v is not None for v in fes_hits),
                "median_fes_to_target": statistics.median([v for v in fes_hits if v is not None])
                if any(v is not None for v in fes_hits)
                else None,
            }
        ranks = rank_values({algo: cell_stats[algo]["mean_error"] for algo in ALGOS})
        best_rank = min(ranks.values())
        for algo, rank in ranks.items():
            rank_lists[algo].append(rank)
            category_rank_lists[category][algo].append(rank)
            if rank == best_rank:
                wins[algo] += 1
            hits[algo] += cell_stats[algo]["hits"]
            cell_stats[algo]["rank"] = rank
        cells[f"F{fid}"] = {"category": category, "stats": cell_stats}

    avg_rank = {algo: statistics.fmean(rank_lists[algo]) for algo in ALGOS}
    category_avg_rank = {
        category: {algo: statistics.fmean(vals) for algo, vals in algo_vals.items()}
        for category, algo_vals in category_rank_lists.items()
    }

    return {
        "settings": {
            "suite": "CEC-2022",
            "dimension": DIM,
            "runs": RUNS,
            "pop_size": POP_SIZE,
            "max_fes": MAX_FES,
            "err_tol": ERR_TOL,
            "algorithms": ALGOS,
            "note": "Mean-error ranks over all CEC-2022 D=10 functions.",
        },
        "average_rank": dict(sorted(avg_rank.items(), key=lambda kv: kv[1])),
        "category_average_rank": category_avg_rank,
        "wins_by_mean_error": dict(wins),
        "hits_to_1e_minus_8": dict(hits),
        "cells": cells,
    }


def main() -> None:
    cache = load_json(CACHE_FILE)
    total = len(FUNCTIONS) * len(ALGOS) * RUNS
    done = len(cache)
    print(f"CEC-2022 D={DIM}, runs={RUNS}, MaxFES={MAX_FES}, algos={ALGOS}")
    print(f"Checkpoint: {done}/{total} runs already complete")

    for fid in FUNCTIONS:
        func = getattr(opfunu.cec_based, f"F{fid}2022")(ndim=DIM)
        f_opt = float(func.f_bias)
        for algo in ALGOS:
            runner = make_runner(algo, fname=f"CEC2022-F{fid}")
            for seed in range(RUNS):
                k = key(fid, algo, seed)
                if k in cache:
                    continue
                t0 = time.time()
                result = runner(
                    func.evaluate,
                    DIM,
                    func.lb,
                    func.ub,
                    POP_SIZE,
                    MAX_FES,
                    ERR_TOL,
                    f_opt,
                    seed,
                )
                elapsed = time.time() - t0
                error = max(0.0, float(result.f_best) - f_opt)
                cache[k] = {
                    "function": f"F{fid}",
                    "category": FUNCTIONS[fid],
                    "algorithm": algo,
                    "seed": seed,
                    "f_best": float(result.f_best),
                    "f_bias": f_opt,
                    "error": error,
                    "fes_to_target": result.fes_to_target,
                    "elapsed_s": elapsed,
                }
                save_json(CACHE_FILE, cache)
                done += 1
                print(
                    f"[{done:03d}/{total}] F{fid:02d} {algo:<6s} seed={seed:<2d} "
                    f"err={error:.3e} hit={result.fes_to_target} time={elapsed:.1f}s",
                    flush=True,
                )

    summary = summarize(cache)
    save_json(SUMMARY_FILE, summary)
    print("\nAverage ranks:")
    for algo, rank in summary["average_rank"].items():
        print(f"  {algo:<6s} {rank:.3f}")
    print(f"\nSaved {CACHE_FILE} and {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
