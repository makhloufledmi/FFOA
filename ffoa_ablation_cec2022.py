"""CEC-2022 mechanism ablation for the canonical FFOA.

This experiment isolates the FFOA mechanisms on rotated/basic,
hybrid, and composition CEC-2022 functions. It is intentionally compact and
checkpointed so it can be resumed.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable

import numpy as np
import opfunu

from ffoa import EvaluationBudgetReached, FFOA


DIM = int(os.environ.get("ABL_DIM", "10"))
RUNS = int(os.environ.get("ABL_RUNS", "5"))
POP_SIZE = int(os.environ.get("ABL_POP_SIZE", "30"))
MAX_FES = int(os.environ.get("ABL_MAX_FES", str(DIM * 3000)))
ERR_TOL = float(os.environ.get("ABL_ERR_TOL", "1e-8"))
TAG = os.environ.get("ABL_TAG", f"d{DIM}_{RUNS}runs_{MAX_FES}fes")
MAX_SECONDS = float(os.environ.get("ABL_MAX_SECONDS", "0"))
MAX_WORKERS = max(1, int(os.environ.get("ABL_WORKERS", "1")))
CACHE_FILE = f"ffoa_ablation_cec2022_{TAG}_results.json"
SUMMARY_FILE = f"ffoa_ablation_cec2022_{TAG}_summary.json"

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

VARIANTS = [
    "Full",
    "NoPreyMemory",
    "NoScentTrail",
    "NoKitMortality",
    "NoPreyMigration",
    "NoCacheEscape",
]


class BudgetReached(Exception):
    pass


class InstrumentedObjective:
    def __init__(self, inner: Callable[[np.ndarray], float], max_fes: int,
                 err_tol: float, f_opt: float):
        self.inner = inner
        self.max_fes = int(max_fes)
        self.err_tol = float(err_tol)
        self.f_opt = float(f_opt)
        self.fes_used = 0
        self.fes_to_target: int | None = None
        self.f_best = math.inf
        self.x_best: np.ndarray | None = None

    def __call__(self, x: np.ndarray) -> float:
        if self.fes_used >= self.max_fes:
            raise BudgetReached
        val = float(self.inner(np.asarray(x, dtype=float)))
        self.fes_used += 1
        if val < self.f_best:
            self.f_best = val
            self.x_best = np.asarray(x, dtype=float).copy()
        if self.fes_to_target is None and (val - self.f_opt) <= self.err_tol:
            self.fes_to_target = self.fes_used
            raise BudgetReached
        return val


@dataclass
class RunResult:
    f_best: float
    fes_to_target: int | None
    fes_used: int


class NoPreyMemoryFFOA(FFOA):
    """Use fixed F/CR values and disable prey-memory adaptation."""

    def _acoustic_pounce(self, i: int, t: int) -> np.ndarray:
        F = 0.5
        CR = 0.5
        p = self._prey_pool_fraction(t)
        top_k = max(1, int(np.ceil(p * self.N)))
        order = np.argsort(self.fitness)[:top_k]
        x_prey = self.X[int(np.random.choice(order))]

        others = [j for j in range(self.N) if j != i]
        r1 = int(np.random.choice(others))
        pool = list(range(self.N)) + [self.N + k for k in range(len(self._archive))]
        pool = [c for c in pool if c != i]
        if pool:
            r2c = int(np.random.choice(pool))
            x_r2 = self.X[r2c] if r2c < self.N else self._archive[r2c - self.N]
        else:
            x_r2 = self.X[r1]

        self._F_used[i] = None
        self._CR_used[i] = None
        v = self.X[i] + F * (x_prey - self.X[i]) + F * (self.X[r1] - x_r2)
        u = self.X[i].copy()
        mask = np.random.rand(self.dim) < CR
        mask[np.random.randint(self.dim)] = True
        u[mask] = v[mask]
        return u


class NoFoodCacheArchiveFFOA(FFOA):
    """Disable the displaced-position food-cache archive."""

    def _archive_insert(self, x: np.ndarray):
        return None

    def _archive_shrink(self, cap: int):
        self._archive_cap = max(1, cap)
        self._archive.clear()


class FixedPopulationFFOA(FFOA):
    """Keep the initial active population size throughout the run."""

    def _population_target_size(self, t: int) -> int:
        return self.N_init

    def _shrink_population(self, target_N: int):
        return None


class NoPreyMigrationFFOA(FFOA):
    """Disable population-level prey-migration dispersal."""

    def _prey_migration_dispersal(self, t: int):
        return None


def make_optimizer(variant: str):
    if variant == "Full":
        return FFOA
    if variant == "NoPreyMemory":
        return NoPreyMemoryFFOA
    if variant == "NoScentTrail":
        return NoFoodCacheArchiveFFOA
    if variant == "NoKitMortality":
        return FixedPopulationFFOA
    if variant == "NoPreyMigration":
        return NoPreyMigrationFFOA
    if variant == "NoCacheEscape":
        return FFOA
    raise ValueError(f"Unknown variant: {variant}")


def run_variant(func, f_opt: float, variant: str, seed: int) -> RunResult:
    instr = InstrumentedObjective(func.evaluate, MAX_FES, ERR_TOL, f_opt)
    pop_min = max(4, POP_SIZE // 5)
    max_iter = max(1, int(math.ceil(MAX_FES / pop_min)) + 1)
    opt_cls = make_optimizer(variant)
    kwargs = {}
    if variant == "NoCacheEscape":
        kwargs.update(patience=10**12, escape_patience=10**12)
    try:
        opt = opt_cls(
            obj_func=instr,
            dim=DIM,
            lb=func.lb,
            ub=func.ub,
            pop_size=POP_SIZE,
            max_iter=max_iter,
            max_fes=MAX_FES,
            pop_min=pop_min,
            seed=seed,
            **kwargs,
        )
        opt.optimize()
    except (BudgetReached, EvaluationBudgetReached):
        pass
    return RunResult(
        f_best=float(instr.f_best),
        fes_to_target=instr.fes_to_target,
        fes_used=instr.fes_used,
    )


def load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def key(fid: int, variant: str, seed: int) -> str:
    return f"F{fid}|{variant}|seed{seed}"


def run_task(task: tuple[int, str, int]) -> tuple[str, dict]:
    fid, variant, seed = task
    func = getattr(opfunu.cec_based, f"F{fid}2022")(ndim=DIM)
    f_opt = float(func.f_bias)
    started = time.time()
    result = run_variant(func, f_opt, variant, seed)
    error = max(0.0, result.f_best - f_opt)
    return key(fid, variant, seed), {
        "function": f"F{fid}",
        "category": FUNCTIONS[fid],
        "variant": variant,
        "seed": seed,
        "f_best": result.f_best,
        "f_bias": f_opt,
        "error": error,
        "fes_to_target": result.fes_to_target,
        "fes_used": result.fes_used,
        "max_fes": MAX_FES,
        "termination": (
            "target" if result.fes_to_target is not None else "budget"
        ),
        "elapsed_s": time.time() - started,
    }


def rank_values(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda kv: kv[1])
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and math.isclose(
            ordered[j][1], ordered[i][1], rel_tol=0.0, abs_tol=1e-12
        ):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for variant, _ in ordered[i:j]:
            ranks[variant] = avg_rank
        i = j
    return ranks


def scored_error(record: dict) -> float:
    error = max(0.0, float(record["error"]))
    return 0.0 if record.get("fes_to_target") is not None or error <= ERR_TOL else error


def summarize(cache: dict) -> dict:
    cells = {}
    ranks = defaultdict(list)
    category_ranks = defaultdict(lambda: defaultdict(list))
    wins = defaultdict(int)
    hits = defaultdict(int)
    deltas_vs_full = defaultdict(list)

    for fid, category in FUNCTIONS.items():
        cell_stats = {}
        full_vals = [
            scored_error(cache[key(fid, "Full", seed)])
            for seed in range(RUNS)
        ]
        full_mean = statistics.fmean(full_vals)
        for variant in VARIANTS:
            vals = [
                scored_error(cache[key(fid, variant, seed)])
                for seed in range(RUNS)
            ]
            fes_hits = [cache[key(fid, variant, seed)]["fes_to_target"] for seed in range(RUNS)]
            mean_error = statistics.fmean(vals)
            cell_stats[variant] = {
                "mean_error": mean_error,
                "median_error": statistics.median(vals),
                "std_error": statistics.pstdev(vals),
                "best_error": min(vals),
                "worst_error": max(vals),
                "hits": sum(v is not None for v in fes_hits),
                "delta_mean_vs_full": mean_error - full_mean,
                "ratio_mean_vs_full": mean_error / (full_mean + 1e-30),
            }
            deltas_vs_full[variant].append(mean_error - full_mean)
            hits[variant] += cell_stats[variant]["hits"]
        cell_ranks = rank_values({v: cell_stats[v]["mean_error"] for v in VARIANTS})
        best_rank = min(cell_ranks.values())
        for variant, rank in cell_ranks.items():
            ranks[variant].append(rank)
            category_ranks[category][variant].append(rank)
            cell_stats[variant]["rank"] = rank
            if rank == best_rank:
                wins[variant] += 1
        cells[f"F{fid}"] = {"category": category, "stats": cell_stats}

    return {
        "settings": {
            "suite": "CEC-2022",
            "dimension": DIM,
            "runs": RUNS,
            "pop_size": POP_SIZE,
            "max_fes": MAX_FES,
            "err_tol": ERR_TOL,
            "final_error_scoring": "target-reaching runs are tied at zero",
            "variants": VARIANTS,
            "note": "Ablation ranks are computed from mean error per CEC-2022 function.",
        },
        "average_rank": dict(sorted(
            {v: statistics.fmean(ranks[v]) for v in VARIANTS}.items(),
            key=lambda kv: kv[1],
        )),
        "category_average_rank": {
            cat: dict(sorted(
                {v: statistics.fmean(vals) for v, vals in by_variant.items()}.items(),
                key=lambda kv: kv[1],
            ))
            for cat, by_variant in category_ranks.items()
        },
        "wins_by_mean_error": dict(wins),
        "hits_to_target": dict(hits),
        "mean_delta_vs_full": {
            v: statistics.fmean(deltas_vs_full[v]) for v in VARIANTS
        },
        "cells": cells,
    }


def main() -> None:
    cache = load_json(CACHE_FILE)
    for record in cache.values():
        record.setdefault("max_fes", MAX_FES)
        record.setdefault(
            "termination",
            "target" if record.get("fes_to_target") is not None else "budget",
        )
    total = len(FUNCTIONS) * len(VARIANTS) * RUNS
    done = len(cache)
    deadline = time.time() + MAX_SECONDS if MAX_SECONDS > 0 else None
    print(
        f"FFOA ablation CEC-2022 D={DIM}, runs={RUNS}, MaxFES={MAX_FES}, "
        f"variants={VARIANTS}, workers={MAX_WORKERS}"
    )
    print(f"Checkpoint: {done}/{total} runs already complete")

    tasks = [
        (fid, variant, seed)
        for fid in FUNCTIONS
        for variant in VARIANTS
        for seed in range(RUNS)
        if key(fid, variant, seed) not in cache
    ]
    if MAX_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            pending = {}
            task_iter = iter(tasks)
            while True:
                while len(pending) < MAX_WORKERS:
                    if deadline is not None and time.time() >= deadline:
                        break
                    try:
                        task = next(task_iter)
                    except StopIteration:
                        break
                    pending[executor.submit(run_task, task)] = task
                if not pending:
                    break
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    pending.pop(future)
                    k, record = future.result()
                    cache[k] = record
                    done += 1
                    save_json(CACHE_FILE, cache)
                    print(
                        f"[{done:04d}/{total}] {record['function']} "
                        f"{record['variant']:<15s} seed={record['seed']:<2d} "
                        f"err={record['error']:.3e} "
                        f"hit={record['fes_to_target']} "
                        f"time={record['elapsed_s']:.1f}s",
                        flush=True,
                    )
                if deadline is not None and time.time() >= deadline:
                    for future in pending:
                        future.cancel()
                    break
        if done < total:
            print(f"Controlled window ended at {done}/{total}")
            return

    for fid in FUNCTIONS:
        if MAX_WORKERS > 1:
            break
        func = getattr(opfunu.cec_based, f"F{fid}2022")(ndim=DIM)
        f_opt = float(func.f_bias)
        for variant in VARIANTS:
            for seed in range(RUNS):
                if deadline is not None and time.time() >= deadline:
                    save_json(CACHE_FILE, cache)
                    print(f"Controlled window ended at {done}/{total}")
                    return
                k = key(fid, variant, seed)
                if k in cache:
                    continue
                t0 = time.time()
                result = run_variant(func, f_opt, variant, seed)
                elapsed = time.time() - t0
                error = max(0.0, result.f_best - f_opt)
                cache[k] = {
                    "function": f"F{fid}",
                    "category": FUNCTIONS[fid],
                    "variant": variant,
                    "seed": seed,
                    "f_best": result.f_best,
                    "f_bias": f_opt,
                    "error": error,
                    "fes_to_target": result.fes_to_target,
                    "fes_used": result.fes_used,
                    "max_fes": MAX_FES,
                    "termination": (
                        "target" if result.fes_to_target is not None else "budget"
                    ),
                    "elapsed_s": elapsed,
                }
                done += 1
                save_json(CACHE_FILE, cache)
                print(
                    f"[{done:03d}/{total}] F{fid:02d} {variant:<15s} "
                    f"seed={seed:<2d} err={error:.3e} hit={result.fes_to_target} "
                    f"time={elapsed:.1f}s",
                    flush=True,
                )

    summary = summarize(cache)
    save_json(SUMMARY_FILE, summary)
    print("\nAverage ranks:")
    for variant, rank in summary["average_rank"].items():
        print(f"  {variant:<15s} {rank:.3f}")
    print("\nMean delta vs Full:")
    for variant, delta in sorted(summary["mean_delta_vs_full"].items(),
                                 key=lambda kv: kv[1]):
        print(f"  {variant:<15s} {delta:.3e}")
    print(f"\nSaved {CACHE_FILE} and {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
