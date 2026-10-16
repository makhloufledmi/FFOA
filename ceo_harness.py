"""
CEO-style benchmark harness — reproduces the experimental protocol of
Dong et al. (2025) §4.1 verbatim:

    - 15 benchmark functions (ceo_benchmarks.BENCHMARKS, Table 1)
    - 5 dimensions: 2, 5, 10, 20, 30
    - both shifted and non-shifted variants (Figs. 7 vs 8)
    - 30 independent runs per cell
    - pop_size = 50 (paper §4.1)
    - MaxFES = dim * 10_000 (paper §4.1)
    - early stop at error ≤ 1e-8 (paper §4.1)
    - per-cell metric: AvgFEvals to reach 1e-8 (or "N/A" if not reached)

Checkpointed JSON, resumable. Run as:

    python ceo_harness.py --algos CEO,FFOA
    python ceo_harness.py --algos all
    python ceo_harness.py --algos LSHADE --funcs Sphere,Ackley --dims 10,30
    python ceo_harness.py --algos all --runs 51 --quick     # 5 seeds for shake-out
    python ceo_harness.py --max-seconds 3600 --algos all    # time-bounded chunked

Cell key format:
    "<suite>|<fname>|D<dim>|<algo>|s<seed>"
where:
    suite ∈ {"unshifted", "shifted"}
    fname is the exact name from ceo_benchmarks.BENCHMARK_BY_NAME
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from ceo_benchmarks import (
    BENCHMARKS,
    BENCHMARK_BY_NAME,
    make_shifted,
    make_unshifted,
    shift_fingerprint,
)
from ceo_comparators import ALL_ALGOS, make_runner


# ──────────────────────────────────────────────────────────────────────
#  Defaults (paper §4.1)
# ──────────────────────────────────────────────────────────────────────

POP_SIZE = 50
ERR_TOL = 1e-8
DIMS_DEFAULT = [2, 5, 10, 20, 30]
SEEDS_DEFAULT = 30

CHECKPOINT = Path("ceo_bench_ckpt.json")
LOG_FILE = Path("ceo_bench.log")
RESULTS = Path("ceo_bench_results.json")

_DEADLINE: float | None = None


# ──────────────────────────────────────────────────────────────────────
#  Persistence
# ──────────────────────────────────────────────────────────────────────


def load_ckpt() -> dict:
    if not CHECKPOINT.exists():
        return {}
    try:
        return json.load(CHECKPOINT.open())
    except json.JSONDecodeError:
        return {}


def save_ckpt(data: dict, max_retries: int = 8, base_delay: float = 0.1):
    """Atomic write with retry on Windows file-lock contention.

    Background sweeps and foreground inspection scripts (figures, stats)
    both read the checkpoint, and Windows refuses os.replace while any
    reader holds the file open. We retry with exponential backoff
    instead of crashing the whole sweep.
    """
    tmp = CHECKPOINT.with_suffix(".json.tmp")
    with tmp.open("w") as fh:
        json.dump(data, fh)
    delay = base_delay
    for attempt in range(max_retries):
        try:
            tmp.replace(CHECKPOINT)
            return
        except PermissionError:
            if attempt == max_retries - 1:
                # Last-ditch: leave the .tmp in place; sweep continues
                # with in-memory state. The next successful save_ckpt
                # will replace both.
                log(f"  [save_ckpt: gave up after {max_retries} retries; "
                    f"data still in memory, tmp file left at {tmp}]")
                return
            time.sleep(delay)
            delay *= 2.0


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def time_up() -> bool:
    return _DEADLINE is not None and time.time() >= _DEADLINE


# ──────────────────────────────────────────────────────────────────────
#  Single-cell execution
# ──────────────────────────────────────────────────────────────────────


def run_cell(suite: str, fname: str, dim: int, algo: str, seed: int,
             data: dict) -> dict:
    """Execute one (suite, fname, dim, algo, seed) cell.

    Returns the stored record. Persists into `data` if newly computed.
    """
    key = f"{suite}|{fname}|D{dim}|{algo}|s{seed}"
    if key in data:
        return data[key]

    b = BENCHMARK_BY_NAME[fname]
    max_fes = dim * 10_000
    if suite == "unshifted":
        fn, lb, ub, f_opt, _ = make_unshifted(b, dim)
        shift_id = None
    elif suite == "shifted":
        fn, lb, ub, f_opt, _ = make_shifted(b, dim)
        shift_id = shift_fingerprint(b, dim)
    else:
        raise ValueError(suite)

    runner = make_runner(algo, fname=fname)
    t0 = time.time()
    try:
        result = runner(fn, dim, lb, ub, POP_SIZE, max_fes, ERR_TOL, f_opt, seed)
        rec = {
            "fes": result.fes_to_target if result.fes_to_target is not None else None,
            "fes_used": result.fes_used,
            "termination": result.termination,
            "fbest": float(result.f_best),
            "time": time.time() - t0,
            "shift_id": shift_id,
        }
    except Exception as exc:
        log(f"  ERROR {key}: {exc}\n{traceback.format_exc()}")
        rec = {"fes": None, "fbest": float("nan"),
               "time": time.time() - t0, "error": str(exc)}
    data[key] = rec
    return rec


# ──────────────────────────────────────────────────────────────────────
#  Sweep driver
# ──────────────────────────────────────────────────────────────────────


def sweep(algos: list[str], funcs: list[str], dims: list[int],
          suites: list[str], runs: int, data: dict) -> bool:
    """Walk through all cells. Returns True if the sweep completed
    without hitting the time budget, False if it was cut short."""
    completed_in_run = 0
    for suite in suites:
        for fname in funcs:
            b = BENCHMARK_BY_NAME[fname]
            for dim in dims:
                for algo in algos:
                    # Skip noisy header if this cell is already complete
                    all_done = all(
                        f"{suite}|{fname}|D{dim}|{algo}|s{s}" in data
                        for s in range(runs)
                    )
                    if not all_done:
                        log(f"  -> {suite}|{fname:<20} D{dim:<3} {algo:<8} ({runs} seeds)")
                    for seed in range(runs):
                        if time_up():
                            save_ckpt(data)
                            log(f"  [time budget hit after {completed_in_run} new cells; checkpoint saved]")
                            return False
                        rec = run_cell(suite, fname, dim, algo, seed, data)
                        completed_in_run += 1
                        # Heartbeat every 17 cells (prime so it spreads across algos)
                        if completed_in_run % 17 == 0:
                            save_ckpt(data)
                            if seed in (0, runs // 2, runs - 1):
                                fes_txt = f"{rec['fes']}" if rec.get('fes') else "N/A"
                                log(f"    {suite}|{fname[:12]}|D{dim}|{algo}|s{seed} "
                                    f"fbest={rec['fbest']:.4e} fes={fes_txt} "
                                    f"({rec['time']:.1f}s)")
                    save_ckpt(data)
    return True


# ──────────────────────────────────────────────────────────────────────
#  Reporting — AvgFEvals to reach 1e-8 (paper §4.1 metric)
# ──────────────────────────────────────────────────────────────────────


def compute_avg_fes(data: dict, suite: str, fname: str, dim: int,
                    algo: str, runs: int) -> tuple[float | None, int, int]:
    """Return (avg_fes_when_reached, n_reached, n_total) for the cell."""
    fes_vals = []
    n_total = 0
    for s in range(runs):
        key = f"{suite}|{fname}|D{dim}|{algo}|s{s}"
        if key not in data:
            continue
        n_total += 1
        v = data[key].get("fes")
        if v is not None:
            fes_vals.append(int(v))
    if not fes_vals:
        return None, 0, n_total
    return float(np.mean(fes_vals)), len(fes_vals), n_total


def summarise(data: dict, algos: list[str], funcs: list[str],
              dims: list[int], suites: list[str], runs: int) -> dict:
    out: dict = {}
    for suite in suites:
        out[suite] = {}
        for fname in funcs:
            out[suite][fname] = {}
            for dim in dims:
                out[suite][fname][f"D{dim}"] = {}
                for algo in algos:
                    avg, hit, total = compute_avg_fes(data, suite, fname, dim, algo, runs)
                    # Also store best/mean/worst fbest across seeds (for funcs
                    # nobody reaches 1e-8 — like Rosenbrock at high D).
                    fbest_vals = []
                    for s in range(runs):
                        key = f"{suite}|{fname}|D{dim}|{algo}|s{s}"
                        if key in data and not np.isnan(data[key].get("fbest", np.nan)):
                            fbest_vals.append(float(data[key]["fbest"]))
                    rec = {
                        "avg_fes": avg,
                        "n_reached": hit,
                        "n_total": total,
                    }
                    if fbest_vals:
                        rec["fbest_mean"] = float(np.mean(fbest_vals))
                        rec["fbest_median"] = float(np.median(fbest_vals))
                        rec["fbest_best"] = float(np.min(fbest_vals))
                        rec["fbest_worst"] = float(np.max(fbest_vals))
                        rec["fbest_std"] = float(np.std(fbest_vals))
                    out[suite][fname][f"D{dim}"][algo] = rec
    return out


def print_table(summary: dict, algos: list[str], suite: str, dim: int,
                runs: int) -> None:
    """Print a paper-style Fig.7/8 table: rows=functions, cols=algos.
    Cell value is avg FEs (or '--' if not reached). ASCII-safe so it
    renders on the Windows cp1252 console."""
    sep = "-" * (24 + 13 * len(algos))
    print("\n" + sep)
    print(f"  {suite.upper()}  D={dim}   (AvgFEvals to reach err <= 1e-8; -- = not reached)")
    print(sep)
    header = "  Function" + " " * 14
    for a in algos:
        header += f"{a:>13}"
    print(header)
    print(sep)
    for fname in summary[suite].keys():
        row = f"  {fname:<22}"
        for a in algos:
            rec = summary[suite][fname][f"D{dim}"].get(a, {})
            avg = rec.get("avg_fes")
            if avg is None:
                row += f"{'--':>13}"
            else:
                row += f"{avg:>13,.0f}"
        print(row)
    print(sep)


# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CEO-style benchmark harness")
    p.add_argument("--algos", default="CEO,FFOA",
                   help="Comma-separated list of algorithms. 'all' = the full 15.")
    p.add_argument("--funcs", default="all",
                   help="Comma-separated benchmark names from Table 1, or 'all'.")
    p.add_argument("--dims", default="2,5,10,20,30",
                   help="Comma-separated list of dimensions.")
    p.add_argument("--suites", default="unshifted,shifted",
                   help="Subset of {unshifted, shifted}.")
    p.add_argument("--runs", type=int, default=SEEDS_DEFAULT,
                   help="Seeds per cell (paper uses 51).")
    p.add_argument("--quick", action="store_true",
                   help="Shorthand for --runs 5 and the 3-protagonist subset.")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="Time-bounded execution; checkpoint and exit when reached.")
    p.add_argument("--report-only", action="store_true",
                   help="Skip the sweep and just print tables from the checkpoint.")
    p.add_argument("--checkpoint", default=str(CHECKPOINT),
                   help="Path to the checkpoint JSON.")
    p.add_argument("--results", default=str(RESULTS),
                   help="Path to the summary JSON output.")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    global _DEADLINE, CHECKPOINT, RESULTS

    args = parse_args(argv)
    CHECKPOINT = Path(args.checkpoint)
    RESULTS = Path(args.results)
    if args.max_seconds is not None:
        _DEADLINE = time.time() + float(args.max_seconds)

    if args.algos == "all":
        algos = list(ALL_ALGOS)
    else:
        algos = [a.strip() for a in args.algos.split(",") if a.strip()]

    if args.funcs == "all":
        funcs = [b.name for b in BENCHMARKS]
    else:
        funcs = [f.strip() for f in args.funcs.split(",") if f.strip()]
        for f in funcs:
            if f not in BENCHMARK_BY_NAME:
                raise SystemExit(f"Unknown benchmark name: {f}")

    dims = [int(d) for d in args.dims.split(",") if d.strip()]
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]

    runs = args.runs
    if args.quick:
        runs = 5
        algos = [a for a in algos if a in ("CEO", "FFOA")] or ["CEO", "FFOA"]

    log(f"ceo_harness  algos={algos}  funcs={len(funcs)}  dims={dims}  "
        f"suites={suites}  runs={runs}")
    data = load_ckpt()
    log(f"  Checkpoint {CHECKPOINT} -> {len(data)} cells")

    if not args.report_only:
        finished = sweep(algos, funcs, dims, suites, runs, data)
    else:
        finished = True

    summary = summarise(data, algos, funcs, dims, suites, runs)
    json.dump(summary, RESULTS.open("w"), indent=1)
    log(f"Summary -> {RESULTS}")

    for suite in suites:
        for dim in dims:
            print_table(summary, algos, suite, dim, runs)

    log("Done." if finished else "Partial (re-run to resume).")
    return 0 if finished else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
