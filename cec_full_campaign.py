r"""Full CEC validation campaign for the canonical FFOA manuscript.

This runner is intentionally checkpointed after every seed so that the
full SWEVO-grade validation can be resumed safely.  By default it runs:

* CEC-2022 at D=10 and D=20, functions F1--F12;
* CEC-2017 at D=10 and D=30, available functions F1--F30;
* 30 independent seeds;
* MaxFES = 10,000 * D;
* a broad comparator slate including FFOA, CEO, adaptive DE variants,
  classical swarm/evolutionary baselines, and recent metaphor algorithms.

The script requires the local dependency path used in this workspace:
    set PYTHONPATH=D:\FFOA.vF\.codex\pydeps

Environment overrides:
    CEC_SUITES=2022,2017
    CEC2022_DIMS=10,20
    CEC2017_DIMS=10,30
    CEC_RUNS=30
    CEC_ALGOS=FFOA,CEO,LSHADE,SADE,DE,PSO,GWO,HHO,SMA,RUN,ABC,WOA,SCA,SSA,MFO,AOA,AVOA
    CEC_TAG=full
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Callable

import numpy as np
import opfunu

from cec_functions import make_cec_function
from ceo_comparators import (
    RunResult,
    _run_mealpy,
    make_runner as make_ceo_runner,
)

from mealpy.math_based.AOA import OriginalAOA
from mealpy.math_based.SCA import OriginalSCA
from mealpy.swarm_based.AVOA import OriginalAVOA
from mealpy.swarm_based.MFO import OriginalMFO
from mealpy.swarm_based.SSA import OriginalSSA
from mealpy.swarm_based.WOA import OriginalWOA


RUNS = int(os.environ.get("CEC_RUNS", "30"))
POP_SIZE = int(os.environ.get("CEC_POP_SIZE", "30"))
ERR_TOL = float(os.environ.get("CEC_ERR_TOL", "1e-8"))
LOG_EVERY = max(1, int(os.environ.get("CEC_LOG_EVERY", "1")))
MAX_WORKERS = max(1, int(os.environ.get("CEC_WORKERS", "1")))
MAX_SECONDS = float(os.environ.get("CEC_MAX_SECONDS", "0"))
SAVE_EVERY = max(1, int(os.environ.get("CEC_SAVE_EVERY", "1")))
SUMMARY_EVERY = max(1, int(os.environ.get("CEC_SUMMARY_EVERY", "10")))
NONFINITE_ERROR_PENALTY = float(os.environ.get("CEC_NONFINITE_ERROR_PENALTY", "1e100"))
TAG = os.environ.get("CEC_TAG", "full")
OUT_DIR = Path(os.environ.get("CEC_OUT_DIR", "."))
CACHE_FILE = OUT_DIR / f"cec_full_{TAG}_results.json"
SUMMARY_FILE = OUT_DIR / f"cec_full_{TAG}_summary.json"

SUITES = [s.strip() for s in os.environ.get("CEC_SUITES", "2022,2017").split(",") if s.strip()]
SUITE_DIMS = {
    "2022": [int(x) for x in os.environ.get("CEC2022_DIMS", "10,20").split(",") if x.strip()],
    "2017": [int(x) for x in os.environ.get("CEC2017_DIMS", "10,30").split(",") if x.strip()],
}
ALGORITHMS = [
    a.strip()
    for a in os.environ.get(
        "CEC_ALGOS",
        "FFOA,CEO,LSHADE,SADE,DE,PSO,GWO,HHO,SMA,RUN,ABC,WOA,SCA,SSA,MFO,AOA,AVOA",
    ).split(",")
    if a.strip()
]

CEC2022_CATEGORIES = {
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


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def cec_function_ids(suite: str) -> list[int]:
    if suite == "2022":
        return parse_int_list(os.environ.get("CEC2022_FUNCS", "1,2,3,4,5,6,7,8,9,10,11,12"))
    if suite == "2017":
        ids = []
        requested = parse_int_list(os.environ.get("CEC2017_FUNCS", ",".join(str(i) for i in range(1, 31))))
        for fid in requested:
            if hasattr(opfunu.cec_based, f"F{fid}2017"):
                ids.append(fid)
        return ids
    raise ValueError(f"Unsupported CEC suite: {suite}")


def make_function(suite: str, fid: int, dim: int):
    return make_cec_function(suite, fid, dim)


def category(suite: str, fid: int) -> str:
    if suite == "2022":
        return CEC2022_CATEGORIES[fid]
    if suite == "2017":
        if fid <= 3:
            return "unimodal/basic"
        if fid <= 10:
            return "multimodal/basic"
        if fid <= 20:
            return "hybrid"
        return "composition"
    return "unknown"


def load_json(path: Path) -> dict:
    candidates = [path, path.with_suffix(path.suffix + ".tmp")]
    valid = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as fh:
                valid.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            continue
    return max(valid, key=len) if valid else {}


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    for attempt in range(10):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                print(f"Checkpoint replace blocked; preserved newer data in {tmp}")
                return
            time.sleep(1.0)


def key(suite: str, dim: int, fid: int, algo: str, seed: int) -> str:
    return f"CEC-{suite}|D{dim}|F{fid}|{algo}|seed{seed}"


def finite_error(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return NONFINITE_ERROR_PENALTY
    return max(0.0, value)


def scored_error(record: dict) -> float:
    error = finite_error(record["error"])
    return 0.0 if record.get("fes_to_target") is not None or error <= ERR_TOL else error


def make_runner(algo: str) -> Callable:
    name = algo.upper().replace("-", "")
    if name == "FFOA":
        return make_ceo_runner("FFOA")
    if name in {"CEO", "LSHADE", "SADE", "DE", "PSO", "GWO", "HHO", "SMA", "RUN", "ABC"}:
        return make_ceo_runner(algo)
    if name == "WOA":
        return lambda *a, **k: _run_mealpy(OriginalWOA, *a, **k)
    if name == "SCA":
        return lambda *a, **k: _run_mealpy(OriginalSCA, *a, **k)
    if name == "SSA":
        return lambda *a, **k: _run_mealpy(OriginalSSA, *a, **k)
    if name == "MFO":
        return lambda *a, **k: _run_mealpy(OriginalMFO, *a, **k)
    if name == "AOA":
        return lambda *a, **k: _run_mealpy(OriginalAOA, *a, **k)
    if name == "AVOA":
        return lambda *a, **k: _run_mealpy(OriginalAVOA, *a, **k)
    raise ValueError(f"Unsupported algorithm: {algo}")


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


def summarize(cache: dict) -> dict:
    cells = {}
    rank_lists = defaultdict(list)
    category_rank_lists = defaultdict(lambda: defaultdict(list))
    suite_dim_rank_lists = defaultdict(lambda: defaultdict(list))
    hits = defaultdict(int)
    wins = defaultdict(int)

    grouped = defaultdict(lambda: defaultdict(dict))
    for k, rec in cache.items():
        suite = str(rec["suite"]).replace("CEC-", "")
        cell = (suite, rec["dimension"], rec["function_id"])
        grouped[cell][rec["algorithm"]][rec["seed"]] = rec

    for (suite, dim, fid), algo_seed_records in sorted(grouped.items()):
        if any(
            algo not in algo_seed_records
            or any(seed not in algo_seed_records[algo] for seed in range(RUNS))
            for algo in ALGORITHMS
        ):
            continue
        cell_stats = {}
        for algo in ALGORITHMS:
            seed_records = [algo_seed_records[algo][seed] for seed in range(RUNS)]
            errors = [scored_error(r) for r in seed_records]
            fes_hits = [r["fes_to_target"] for r in seed_records]
            cell_stats[algo] = {
                "mean_error": statistics.fmean(errors),
                "median_error": statistics.median(errors),
                "std_error": statistics.pstdev(errors),
                "best_error": min(errors),
                "worst_error": max(errors),
                "hits": sum(v is not None for v in fes_hits),
                "median_fes_to_target": statistics.median([v for v in fes_hits if v is not None])
                if any(v is not None for v in fes_hits)
                else None,
            }

        ranks = rank_values({algo: cell_stats[algo]["mean_error"] for algo in ALGORITHMS})
        best_rank = min(ranks.values())
        for algo, rank in ranks.items():
            rank_lists[algo].append(rank)
            suite_dim_rank_lists[f"CEC-{suite}_D{dim}"][algo].append(rank)
            category_rank_lists[f"CEC-{suite}_D{dim}_{category(suite, fid)}"][algo].append(rank)
            if rank == best_rank:
                wins[algo] += 1
            hits[algo] += cell_stats[algo]["hits"]
            cell_stats[algo]["rank"] = rank
        cells[f"CEC-{suite}|D{dim}|F{fid}"] = {
            "suite": f"CEC-{suite}",
            "dimension": dim,
            "function": f"F{fid}",
            "category": category(suite, fid),
            "stats": cell_stats,
        }

    def avg_rank(rank_map):
        return {algo: statistics.fmean(vals) for algo, vals in rank_map.items()}

    return {
        "settings": {
            "suites": SUITES,
            "suite_dims": SUITE_DIMS,
            "runs": RUNS,
            "pop_size": POP_SIZE,
            "err_tol": ERR_TOL,
            "final_error_scoring": "target-reaching runs are tied at zero",
            "algorithms": ALGORITHMS,
            "max_fes_rule": "10000 * dimension",
        },
        "average_rank": dict(sorted(avg_rank(rank_lists).items(), key=lambda kv: kv[1])),
        "suite_dimension_average_rank": {
            key_: dict(sorted(avg_rank(vals).items(), key=lambda kv: kv[1]))
            for key_, vals in suite_dim_rank_lists.items()
        },
        "category_average_rank": {
            key_: dict(sorted(avg_rank(vals).items(), key=lambda kv: kv[1]))
            for key_, vals in category_rank_lists.items()
        },
        "wins_by_mean_error": dict(wins),
        "hits_to_1e_minus_8": dict(hits),
        "cells": cells,
    }


def expected_keys() -> list[str]:
    keys = []
    for suite in SUITES:
        for dim in SUITE_DIMS[suite]:
            for fid in cec_function_ids(suite):
                for algo in ALGORITHMS:
                    for seed in range(RUNS):
                        keys.append(key(suite, dim, fid, algo, seed))
    return keys


def expected_tasks() -> list[tuple[str, int, int, str, int]]:
    tasks = []
    for suite in SUITES:
        for dim in SUITE_DIMS[suite]:
            for fid in cec_function_ids(suite):
                for algo in ALGORITHMS:
                    for seed in range(RUNS):
                        tasks.append((suite, dim, fid, algo, seed))
    return tasks


def run_task(task: tuple[str, int, int, str, int]) -> tuple[str, dict]:
    suite, dim, fid, algo, seed = task
    max_fes = dim * 10_000
    func = make_function(suite, fid, dim)
    lb = np.asarray(func.lb, dtype=float)
    ub = np.asarray(func.ub, dtype=float)
    f_opt = float(func.f_bias)
    runner = make_runner(algo)
    t0 = time.time()
    result: RunResult = runner(
        func.evaluate,
        dim,
        lb,
        ub,
        POP_SIZE,
        max_fes,
        ERR_TOL,
        f_opt,
        seed,
    )
    elapsed = time.time() - t0
    f_best = float(result.f_best)
    error = finite_error(f_best - f_opt)
    return key(suite, dim, fid, algo, seed), {
        "suite": f"CEC-{suite}",
        "dimension": dim,
        "function_id": fid,
        "function": f"F{fid}",
        "category": category(suite, fid),
        "algorithm": algo,
        "seed": seed,
        "pop_size": POP_SIZE,
        "max_fes": max_fes,
        "f_best": f_best,
        "f_bias": f_opt,
        "error": error,
        "fes_to_target": result.fes_to_target,
        "fes_used": result.fes_used,
        "termination": result.termination,
        "elapsed_s": elapsed,
    }


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".codex") / "matplotlib"))
    cache = load_json(CACHE_FILE)
    keys = expected_keys()
    print(
        f"CEC full campaign: suites={SUITES}, dims={SUITE_DIMS}, runs={RUNS}, "
        f"algos={ALGORITHMS}, workers={MAX_WORKERS}"
    )
    print(f"Checkpoint: {sum(1 for k in keys if k in cache)}/{len(keys)} complete")
    print(f"Results: {CACHE_FILE}")

    done = sum(1 for k in keys if k in cache)
    deadline = time.time() + MAX_SECONDS if MAX_SECONDS > 0 else None

    if MAX_WORKERS > 1:
        tasks = [task for task in expected_tasks() if key(*task) not in cache]
        pending_limit = max(MAX_WORKERS * 2, 1)
        task_iter = iter(tasks)
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            pending = set()
            for _ in range(min(pending_limit, len(tasks))):
                pending.add(executor.submit(run_task, next(task_iter)))

            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    k, rec = future.result()
                    cache[k] = rec
                    done += 1
                    if done % SAVE_EVERY == 0 or done == len(keys):
                        save_json(CACHE_FILE, cache)
                    if done % SUMMARY_EVERY == 0 or done == len(keys):
                        save_json(SUMMARY_FILE, summarize(cache))
                    if done % LOG_EVERY == 0 or done == len(keys):
                        print(
                            f"[{done:05d}/{len(keys)}] {rec['suite']} D={rec['dimension']} "
                            f"F{rec['function_id']:02d} {rec['algorithm']:<7s} "
                            f"seed={rec['seed']:<2d} err={rec['error']:.3e} "
                            f"hit={rec['fes_to_target']} time={rec['elapsed_s']:.1f}s",
                            flush=True,
                        )
                    if deadline is None or time.time() < deadline:
                        try:
                            pending.add(executor.submit(run_task, next(task_iter)))
                        except StopIteration:
                            pass

        summary = summarize(cache)
        save_json(SUMMARY_FILE, summary)
        if done < len(keys):
            print(f"\nControlled window ended at {done}/{len(keys)} records.")
            return
        print("\nOverall average ranks:")
        for algo, rank in summary["average_rank"].items():
            print(f"  {algo:<7s} {rank:.3f}")
        print(f"\nSaved {CACHE_FILE} and {SUMMARY_FILE}")
        return

    runners = {algo: make_runner(algo) for algo in ALGORITHMS}

    for suite in SUITES:
        for dim in SUITE_DIMS[suite]:
            max_fes = dim * 10_000
            for fid in cec_function_ids(suite):
                func = make_function(suite, fid, dim)
                lb = np.asarray(func.lb, dtype=float)
                ub = np.asarray(func.ub, dtype=float)
                f_opt = float(func.f_bias)
                for algo in ALGORITHMS:
                    runner = runners[algo]
                    for seed in range(RUNS):
                        if deadline is not None and time.time() >= deadline:
                            save_json(CACHE_FILE, cache)
                            save_json(SUMMARY_FILE, summarize(cache))
                            print(f"Controlled window ended at {done}/{len(keys)} records.")
                            return
                        k = key(suite, dim, fid, algo, seed)
                        if k in cache:
                            continue
                        t0 = time.time()
                        result: RunResult = runner(
                            func.evaluate,
                            dim,
                            lb,
                            ub,
                            POP_SIZE,
                            max_fes,
                            ERR_TOL,
                            f_opt,
                            seed,
                        )
                        elapsed = time.time() - t0
                        f_best = float(result.f_best)
                        error = finite_error(f_best - f_opt)
                        cache[k] = {
                            "suite": f"CEC-{suite}",
                            "dimension": dim,
                            "function_id": fid,
                            "function": f"F{fid}",
                            "category": category(suite, fid),
                            "algorithm": algo,
                            "seed": seed,
                            "pop_size": POP_SIZE,
                            "max_fes": max_fes,
                            "f_best": f_best,
                            "f_bias": f_opt,
                            "error": error,
                            "fes_to_target": result.fes_to_target,
                            "fes_used": result.fes_used,
                            "termination": result.termination,
                            "elapsed_s": elapsed,
                        }
                        done += 1
                        if done % SAVE_EVERY == 0 or done == len(keys):
                            save_json(CACHE_FILE, cache)
                        if done % SUMMARY_EVERY == 0 or done == len(keys):
                            save_json(SUMMARY_FILE, summarize(cache))
                        if done % LOG_EVERY == 0 or done == len(keys):
                            print(
                                f"[{done:05d}/{len(keys)}] CEC-{suite} D={dim} F{fid:02d} "
                                f"{algo:<7s} seed={seed:<2d} err={error:.3e} "
                                f"hit={result.fes_to_target} time={elapsed:.1f}s",
                                flush=True,
                            )

    summary = summarize(cache)
    save_json(SUMMARY_FILE, summary)
    print("\nOverall average ranks:")
    for algo, rank in summary["average_rank"].items():
        print(f"  {algo:<7s} {rank:.3f}")
    print(f"\nSaved {CACHE_FILE} and {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
