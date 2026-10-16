"""Checkpointed multi-suite minimal-core ablation for canonical FFOA.

Defaults: all available CEC-2017/2022 functions, two native dimensions per
suite, 30 paired seeds, and 3,000 exact objective evaluations per dimension.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import statistics
import sys
import time
import types
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

# opfunu 1.x uses pkg_resources only for resource_filename. Setuptools 83
# removed that legacy module, so provide the one API opfunu needs.
try:
    import pkg_resources  # noqa: F401
except ModuleNotFoundError:
    pkg_resources = types.ModuleType("pkg_resources")

    def _resource_filename(package: str, resource_name: str) -> str:
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            raise ModuleNotFoundError(package)
        return str(Path(next(iter(spec.submodule_search_locations))) / resource_name)

    pkg_resources.resource_filename = _resource_filename
    sys.modules["pkg_resources"] = pkg_resources

import opfunu
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from cec_functions import make_cec_function
from ffoa import EvaluationBudgetReached, FFOA

RUNS = int(os.environ.get("MC_RUNS", "30"))
POP_SIZE = int(os.environ.get("MC_POP_SIZE", "30"))
FES_PER_DIM = int(os.environ.get("MC_FES_PER_DIM", "3000"))
ERR_TOL = float(os.environ.get("MC_ERR_TOL", "1e-8"))
MAX_WORKERS = max(1, int(os.environ.get("MC_WORKERS", "12")))
MAX_SECONDS = float(os.environ.get("MC_MAX_SECONDS", "0"))
LOG_EVERY = max(1, int(os.environ.get("MC_LOG_EVERY", "25")))
SAVE_EVERY = max(1, int(os.environ.get("MC_SAVE_EVERY", "10")))
TAG = os.environ.get("MC_TAG", "30seeds_multidim_bothcec")
OUT_DIR = Path(os.environ.get("MC_OUT_DIR", "."))
RAW_FILE = OUT_DIR / f"ffoa_minimal_core_{TAG}_results.json"
SUMMARY_FILE = OUT_DIR / f"ffoa_minimal_core_{TAG}_summary.json"
REPORT_FILE = OUT_DIR / f"FFOA_MINIMAL_CORE_{TAG}_REPORT.md"
SUITES = [x.strip() for x in os.environ.get("MC_SUITES", "2022,2017").split(",") if x.strip()]
SUITE_DIMS = {
    "2022": [int(x) for x in os.environ.get("MC_2022_DIMS", "10,20").split(",") if x.strip()],
    "2017": [int(x) for x in os.environ.get("MC_2017_DIMS", "10,30").split(",") if x.strip()],
}
VARIANTS = [
    "AcousticPounceOnly", "PounceSuccessMemory", "PounceArchive",
    "PouncePopulationReduction", "CompleteModel",
]


def parse_ids(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def function_ids(suite: str) -> list[int]:
    default = "1,2,3,4,5,6,7,8,9,10,11,12" if suite == "2022" else ",".join(map(str, range(1, 31)))
    requested = parse_ids(os.environ.get(f"MC_{suite}_FUNCS", default))
    return [fid for fid in requested if hasattr(opfunu.cec_based, f"F{fid}{suite}")]


def category(suite: str, fid: int) -> str:
    if suite == "2022":
        return "rotated/basic" if fid <= 5 else "hybrid" if fid <= 8 else "composition"
    return "unimodal/basic" if fid <= 3 else "multimodal/basic" if fid <= 10 else "hybrid" if fid <= 20 else "composition"


class BudgetReached(Exception):
    pass


class InstrumentedObjective:
    def __init__(self, inner, max_fes: int, err_tol: float, f_opt: float):
        self.inner, self.max_fes, self.err_tol, self.f_opt = inner, int(max_fes), float(err_tol), float(f_opt)
        self.fes_used, self.fes_to_target, self.f_best = 0, None, math.inf

    def __call__(self, x: np.ndarray) -> float:
        if self.fes_used >= self.max_fes:
            raise BudgetReached
        value = float(self.inner(np.asarray(x, dtype=float)))
        self.fes_used += 1
        if value < self.f_best:
            self.f_best = value
        if self.fes_to_target is None and value - self.f_opt <= self.err_tol:
            self.fes_to_target = self.fes_used
            raise BudgetReached
        return value


class AlwaysPounceMixin:
    def _acoustic_activity(self, t: int) -> float:
        return 1.0

    def _prey_migration_dispersal(self, t: int):
        return None


class FixedPopulationMixin:
    def _population_target_size(self, t: int) -> int:
        return self.N_init


class NoArchiveMixin:
    def _archive_insert(self, x: np.ndarray):
        return None

    def _archive_shrink(self, cap: int):
        self._archive_cap = max(1, int(cap))
        self._archive.clear()


class FixedPounceMixin:
    use_archive = False

    def _acoustic_pounce(self, i: int, t: int) -> np.ndarray:
        F, CR = 0.5, 0.5
        top_k = max(1, int(np.ceil(self._prey_pool_fraction(t) * self.N)))
        order = np.argsort(self.fitness)[:top_k]
        x_prey = self.X[int(np.random.choice(order))]
        others = [j for j in range(self.N) if j != i]
        r1 = int(np.random.choice(others))
        pool = list(range(self.N))
        if self.use_archive:
            pool += [self.N + k for k in range(len(self._archive))]
        pool = [candidate for candidate in pool if candidate != i]
        r2c = int(np.random.choice(pool))
        x_r2 = self.X[r2c] if r2c < self.N else self._archive[r2c - self.N]
        self._F_used[i] = self._CR_used[i] = None
        mutant = self.X[i] + F * (x_prey - self.X[i]) + F * (self.X[r1] - x_r2)
        trial = self.X[i].copy()
        mask = np.random.rand(self.dim) < CR
        mask[np.random.randint(self.dim)] = True
        trial[mask] = mutant[mask]
        return trial


class AcousticPounceOnly(AlwaysPounceMixin, FixedPopulationMixin, NoArchiveMixin, FixedPounceMixin, FFOA):
    pass


class PounceSuccessMemory(AlwaysPounceMixin, FixedPopulationMixin, NoArchiveMixin, FFOA):
    pass


class PounceArchive(AlwaysPounceMixin, FixedPopulationMixin, FixedPounceMixin, FFOA):
    use_archive = True


class PouncePopulationReduction(AlwaysPounceMixin, NoArchiveMixin, FixedPounceMixin, FFOA):
    pass


OPTIMIZERS = {
    "AcousticPounceOnly": AcousticPounceOnly,
    "PounceSuccessMemory": PounceSuccessMemory,
    "PounceArchive": PounceArchive,
    "PouncePopulationReduction": PouncePopulationReduction,
    "CompleteModel": FFOA,
}


def record_key(suite: str, dim: int, fid: int, variant: str, seed: int) -> str:
    return f"CEC-{suite}|D{dim}|F{fid}|{variant}|seed{seed}"


def expected_tasks() -> list[tuple[str, int, int, str, int]]:
    return [(suite, dim, fid, variant, seed)
            for suite in SUITES for dim in SUITE_DIMS[suite]
            for fid in function_ids(suite) for variant in VARIANTS
            for seed in range(RUNS)]


def run_task(task: tuple[str, int, int, str, int]) -> tuple[str, dict]:
    suite, dim, fid, variant, seed = task
    max_fes = FES_PER_DIM * dim
    func = make_cec_function(suite, fid, dim)
    f_opt = float(func.f_bias)
    objective = InstrumentedObjective(func.evaluate, max_fes, ERR_TOL, f_opt)
    pop_min = max(4, POP_SIZE // 5)
    max_iter = max(1, math.ceil(max_fes / pop_min) + 1)
    kwargs = {}
    if variant != "CompleteModel":
        kwargs.update(stagnation_cache=10**12, stagnation_escape=10**12, restart_patience=10**12)
    started = time.time()
    try:
        optimizer = OPTIMIZERS[variant](
            obj_func=objective, dim=dim, lb=np.asarray(func.lb, dtype=float),
            ub=np.asarray(func.ub, dtype=float), pop_size=POP_SIZE,
            pop_min=pop_min, max_iter=max_iter, max_fes=max_fes,
            seed=seed, **kwargs)
        optimizer.optimize()
    except (BudgetReached, EvaluationBudgetReached):
        pass
    raw_error = float(objective.f_best - f_opt)
    error = max(0.0, raw_error) if math.isfinite(raw_error) else 1e100
    return record_key(*task), {
        "suite": f"CEC-{suite}", "dimension": dim, "function_id": fid,
        "function": f"F{fid}", "category": category(suite, fid),
        "variant": variant, "seed": seed, "max_fes": max_fes,
        "fes_used": objective.fes_used, "fes_to_target": objective.fes_to_target,
        "error": error, "elapsed_s": time.time() - started,
    }


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
    temporary.replace(path)


def rank_biserial(reference: np.ndarray, comparator: np.ndarray) -> float:
    differences = reference - comparator
    differences = differences[differences != 0]
    if not differences.size:
        return 0.0
    ranks = rankdata(np.abs(differences), method="average")
    better = float(np.sum(ranks[differences < 0]))
    worse = float(np.sum(ranks[differences > 0]))
    return (better - worse) / (better + worse)


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted, running, total = {}, 0.0, len(ordered)
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * raw[name]))
        adjusted[name] = running
    return adjusted


def summarize(cache: dict) -> dict:
    grouped = defaultdict(lambda: defaultdict(dict))
    for record in cache.values():
        cell = (record["suite"], record["dimension"], record["function_id"])
        grouped[cell][record["variant"]][record["seed"]] = record
    cells, rank_lists = {}, defaultdict(list)
    stratum_ranks = defaultdict(lambda: defaultdict(list))
    for cell, by_variant in sorted(grouped.items()):
        if any(variant not in by_variant or len(by_variant[variant]) != RUNS for variant in VARIANTS):
            continue
        stats, medians = {}, {}
        for variant in VARIANTS:
            records = [by_variant[variant][seed] for seed in range(RUNS)]
            errors = np.asarray([
                1e-300 if record["fes_to_target"] is not None
                else max(float(record["error"]), 1e-300)
                for record in records
            ])
            logs = np.log10(errors)
            medians[variant] = float(np.median(logs))
            stats[variant] = {
                "median_log10_error": medians[variant],
                "mean_log10_error": float(np.mean(logs)),
                "median_error": float(np.median(errors)),
                "hits": int(sum(record["fes_to_target"] is not None for record in records)),
            }
        ranks = rankdata([medians[variant] for variant in VARIANTS], method="average")
        stratum = f"{cell[0]}_D{cell[1]}"
        for variant, rank in zip(VARIANTS, ranks):
            stats[variant]["rank"] = float(rank)
            rank_lists[variant].append(float(rank))
            stratum_ranks[stratum][variant].append(float(rank))
        cells[f"{cell[0]}|D{cell[1]}|F{cell[2]}"] = {"stats": stats}
    expected_cells = sum(len(function_ids(suite)) * len(SUITE_DIMS[suite]) for suite in SUITES)
    if not cells:
        return {"status": "partial", "complete_cells": 0, "expected_cells": expected_cells,
                "settings": settings(), "average_rank": {}, "comparisons": {}, "cells": {}}
    cell_names = list(cells)
    matrix = np.asarray([[cells[cell]["stats"][variant]["median_log10_error"]
                          for cell in cell_names] for variant in VARIANTS])
    friedman = friedmanchisquare(*matrix)
    contrast_specs = [
        ("complete_vs_pounce_only", "CompleteModel", "AcousticPounceOnly"),
        ("complete_vs_success_memory", "CompleteModel", "PounceSuccessMemory"),
        ("complete_vs_archive", "CompleteModel", "PounceArchive"),
        ("complete_vs_population_reduction", "CompleteModel", "PouncePopulationReduction"),
        ("success_memory_vs_pounce_only", "PounceSuccessMemory", "AcousticPounceOnly"),
        ("archive_vs_pounce_only", "PounceArchive", "AcousticPounceOnly"),
        ("population_reduction_vs_pounce_only", "PouncePopulationReduction", "AcousticPounceOnly"),
    ]
    comparisons, raw_p = {}, {}
    for name, reference, comparator in contrast_specs:
        a, b = matrix[VARIANTS.index(reference)], matrix[VARIANTS.index(comparator)]
        test = wilcoxon(a, b, zero_method="pratt", alternative="two-sided")
        raw_p[name] = float(test.pvalue)
        comparisons[name] = {
            "reference": reference, "comparator": comparator, "cells": len(cell_names),
            "reference_better": int(np.sum(a < b)), "ties": int(np.sum(a == b)),
            "reference_worse": int(np.sum(a > b)), "wilcoxon_statistic": float(test.statistic),
            "p_raw": float(test.pvalue),
            "rank_biserial_favorable_to_reference": rank_biserial(a, b),
        }
    adjusted = holm_adjust(raw_p)
    for name, result in comparisons.items():
        result["p_holm"] = adjusted[name]
        result["significant_0_05"] = adjusted[name] < 0.05
    return {
        "status": "complete" if len(cells) == expected_cells else "partial",
        "complete_cells": len(cells), "expected_cells": expected_cells,
        "settings": settings(),
        "friedman": {"statistic": float(friedman.statistic), "p_value": float(friedman.pvalue)},
        "average_rank": dict(sorted({variant: statistics.fmean(rank_lists[variant])
                                     for variant in VARIANTS}.items(), key=lambda item: item[1])),
        "suite_dimension_average_rank": {
            stratum: dict(sorted({variant: statistics.fmean(values[variant])
                                 for variant in VARIANTS}.items(), key=lambda item: item[1]))
            for stratum, values in stratum_ranks.items()},
        "comparisons": comparisons, "cells": cells,
    }


def settings() -> dict:
    return {
        "suites": SUITES, "suite_dimensions": SUITE_DIMS,
        "functions": {suite: function_ids(suite) for suite in SUITES},
        "runs_per_cell": RUNS, "population": POP_SIZE,
        "fes_per_dimension": FES_PER_DIM, "variants": VARIANTS,
        "inference_unit": "per-cell median log10 final error over paired seeds",
        "holm_family": "all seven prespecified minimal-core contrasts",
    }


def write_report(summary: dict) -> None:
    lines = ["# FFOA Multi-Suite Minimal-Core Ablation", "",
             f"Status: **{summary['status']}**. Complete cells: {summary['complete_cells']}/{summary['expected_cells']}; paired seeds per cell: {RUNS}; exact budget: {FES_PER_DIM} evaluations per dimension.", "",
             "## Average ranks", "", "| Variant | Average rank |", "|---|---:|"]
    lines.extend(f"| {variant} | {rank:.3f} |" for variant, rank in summary["average_rank"].items())
    lines.extend(["", "## Planned paired contrasts", "",
                  "Positive rank-biserial values favor the named reference. W/T/L is reference better/tied/worse across cell medians.", "",
                  "| Contrast | Reference | Comparator | W/T/L | p raw | p Holm | r_rb |",
                  "|---|---|---|---:|---:|---:|---:|"])
    for name, result in summary["comparisons"].items():
        wtl = f"{result['reference_better']}/{result['ties']}/{result['reference_worse']}"
        lines.append(f"| {name} | {result['reference']} | {result['comparator']} | {wtl} | {result['p_raw']:.3g} | {result['p_holm']:.3g} | {result['rank_biserial_favorable_to_reference']:.3f} |")
    lines.extend(["", "## Statistical design", "",
                  "The inference unit is the suite-dimension-function cell, summarized by median log10 final error over common seeds. A Friedman test covers all five variants. Seven prespecified two-sided Wilcoxon signed-rank contrasts use Pratt zero handling and one Holm family; rank-biserial effects and W/T/L counts retain direction and magnitude.", ""])
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cache = load_json(RAW_FILE)
    tasks = expected_tasks()
    todo = [task for task in tasks if record_key(*task) not in cache]
    done = len(tasks) - len(todo)
    deadline = time.time() + MAX_SECONDS if MAX_SECONDS > 0 else None
    print(f"Minimal-core ablation: {done}/{len(tasks)} complete; workers={MAX_WORKERS}; output={RAW_FILE}")
    if MAX_WORKERS > 1 and todo:
        iterator = iter(todo)
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            pending = set()
            for _ in range(min(MAX_WORKERS * 2, len(todo))):
                pending.add(executor.submit(run_task, next(iterator)))
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    key, record = future.result()
                    cache[key] = record
                    done += 1
                    if done % SAVE_EVERY == 0 or done == len(tasks):
                        save_json(RAW_FILE, cache)
                    if done % LOG_EVERY == 0 or done == len(tasks):
                        print(f"[{done:05d}/{len(tasks)}] {record['suite']} D={record['dimension']} {record['function']} {record['variant']} seed={record['seed']} err={record['error']:.3e} time={record['elapsed_s']:.2f}s", flush=True)
                    if deadline is None or time.time() < deadline:
                        try:
                            pending.add(executor.submit(run_task, next(iterator)))
                        except StopIteration:
                            pass
        save_json(RAW_FILE, cache)
    elif todo:
        for task in todo:
            if deadline is not None and time.time() >= deadline:
                break
            key, record = run_task(task)
            cache[key] = record
            done += 1
            if done % SAVE_EVERY == 0 or done == len(tasks):
                save_json(RAW_FILE, cache)
    save_json(RAW_FILE, cache)
    summary = summarize(cache)
    save_json(SUMMARY_FILE, summary)
    if summary["complete_cells"]:
        write_report(summary)
    print(f"Completed records: {len(cache)}/{len(tasks)}; complete cells: {summary['complete_cells']}; status={summary['status']}")
    print(f"Saved {RAW_FILE}, {SUMMARY_FILE}, and {REPORT_FILE}")


if __name__ == "__main__":
    main()