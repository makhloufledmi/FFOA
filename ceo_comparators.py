"""
Comparator algorithms for the CEO benchmark study (Dong et al. 2025, §4).

This module wraps mealpy implementations of the 12 algorithms used in the
CEO paper, plus a minimal real-coded GL-25 (not in mealpy), behind a
single uniform interface:

    runner(obj_func, dim, lb, ub, pop_size, max_fes, err_tol, f_opt, seed)
        -> RunResult(fes_to_target, f_best, x_best)

The wrapper instruments `obj_func` so that:
  - every call is counted (giving FE-accurate metrics);
  - if (f - f_opt) ≤ err_tol, a sentinel exception is raised and caught,
    short-circuiting the inner mealpy loop — this matches §4.1 of the
    paper which terminates each run at MaxFES OR error ≤ 1e-8.

The paper's 12 comparators are:
  GWO, HHO, SMA, RUN, ABC, RCGA, PSO, DE, CLPSO, GL-25, SaDE, LSHADE.

mealpy 3.0.3 provides direct implementations of all of these except
RCGA and GL-25. We use mealpy's `OriginalGA` (a real-coded GA with
binary tournament + arithmetic crossover + Gaussian mutation) as the
RCGA proxy; this matches the standard textbook RCGA used in most
benchmark studies. GL-25 (García-Martínez, Lozano, Herrera 2008) is
implemented from scratch — it's a real-coded GA that switches between
two phases (global PBX-α with 25% bias toward male parents, then local
BLX-α restricted around the best 25% of the population).

Note on "fairness": each algorithm uses the parameter set specified by
its mealpy default OR its original paper, whichever is documented in
mealpy's docstring — this matches the paper's claim (§4.1) that
"the detailed parameters of the comparison algorithms are kept
consistent with their respective original studies".
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable

# Silence mealpy's logger
import logging
logging.getLogger("mealpy").setLevel(logging.CRITICAL)

from mealpy import FloatVar
from mealpy.evolutionary_based.DE import OriginalDE, SADE
from mealpy.evolutionary_based.GA import BaseGA
from mealpy.evolutionary_based.SHADE import L_SHADE
from mealpy.swarm_based.PSO import OriginalPSO, CL_PSO
from mealpy.swarm_based.GWO import OriginalGWO
from mealpy.swarm_based.HHO import OriginalHHO
from mealpy.swarm_based.ABC import OriginalABC
from mealpy.bio_based.SMA import OriginalSMA
from mealpy.math_based.RUN import OriginalRUN

# Our own implementations
from ceo import CEO
from ffoa import EvaluationBudgetReached, FFOA


@dataclass
class RunResult:
    fes_to_target: int | None   # FE count at which err_tol was reached, else None
    f_best: float               # Best function value seen
    x_best: np.ndarray          # Best solution seen
    fes_used: int               # Total objective calls consumed
    termination: str            # "target", "budget", or "epoch_limit"


class _BudgetReached(Exception):
    """Raised by the instrumented objective to short-circuit the inner loop."""


class _InstrumentedObjective:
    """Wrap an objective with an FE counter + early-stop on target.

    The wrapped callable transparently counts every evaluation, tracks
    the best-so-far solution, and raises `_BudgetReached` either when
    the FE budget is exhausted or when the error to the optimum drops
    below `err_tol`. The host runner catches the exception and reads
    `self.fes_to_target` / `self.f_best` / `self.x_best`.
    """

    __slots__ = ("inner", "max_fes", "err_tol", "f_opt",
                 "fes_used", "fes_to_target", "f_best", "x_best",
                 "_stop_on_target")

    def __init__(self, inner: Callable[[np.ndarray], float],
                 max_fes: int, err_tol: float, f_opt: float,
                 stop_on_target: bool = True):
        self.inner = inner
        self.max_fes = int(max_fes)
        self.err_tol = float(err_tol)
        self.f_opt = float(f_opt)
        self.fes_used = 0
        self.fes_to_target: int | None = None
        self.f_best = np.inf
        self.x_best: np.ndarray | None = None
        self._stop_on_target = bool(stop_on_target)

    def __call__(self, x: np.ndarray) -> float:
        if self.fes_used >= self.max_fes:
            # Hard cap: return a sentinel value; some optimizers will
            # treat this as "extremely bad" and skip it. Combined with
            # the _BudgetReached exception below, this ensures no
            # algorithm ever exceeds its FE budget.
            raise _BudgetReached
        val = float(self.inner(np.asarray(x, dtype=float)))
        self.fes_used += 1
        if val < self.f_best:
            self.f_best = val
            self.x_best = np.asarray(x, dtype=float).copy()
        if self.fes_to_target is None and (val - self.f_opt) <= self.err_tol:
            self.fes_to_target = self.fes_used
            if self._stop_on_target:
                raise _BudgetReached
        return val


def _make_result(instr: _InstrumentedObjective, dim: int,
                 termination_override: str | None = None) -> RunResult:
    if termination_override is not None:
        termination = termination_override
    elif instr.fes_to_target is not None:
        termination = "target"
    elif instr.fes_used == instr.max_fes:
        termination = "budget"
    else:
        termination = "epoch_limit"
    return RunResult(
        fes_to_target=instr.fes_to_target,
        f_best=instr.f_best,
        x_best=(instr.x_best if instr.x_best is not None
                else np.full(dim, np.nan)),
        fes_used=instr.fes_used,
        termination=termination,
    )


def _run_mealpy(algo_cls, obj_func, dim, lb, ub, pop_size, max_fes,
                err_tol, f_opt, seed, **algo_kwargs) -> RunResult:
    """Run a mealpy algorithm under the instrumented objective."""
    instr = _InstrumentedObjective(obj_func, max_fes, err_tol, f_opt)
    problem = {
        "obj_func": instr,
        "bounds": FloatVar(lb=tuple(float(v) for v in np.broadcast_to(lb, dim)),
                           ub=tuple(float(v) for v in np.broadcast_to(ub, dim))),
        "minmax": "min",
        "log_to": None,
    }
    # Epoch budget: enough to consume all FEs; mealpy will run until the
    # objective raises _BudgetReached or it completes the epoch count.
    epoch = max(1, int(np.ceil(max_fes / max(1, pop_size))) + 2)
    model = algo_cls(epoch=epoch, pop_size=pop_size, **algo_kwargs)
    try:
        model.solve(problem, seed=seed)
    except _BudgetReached:
        pass
    return _make_result(instr, dim)


# ──────────────────────────────────────────────────────────────────────
#  GL-25 implementation (García-Martínez, Lozano, Herrera 2008)
# ──────────────────────────────────────────────────────────────────────
#
# A real-coded GA with two interlocked search phases:
#   - Global phase (25% of fitness evaluations): PBX-α crossover with a
#     male/female asymmetric selection that biases offspring toward the
#     fittest 25% of the population (the "alpha males").
#   - Local phase (75% of evaluations): BLX-α crossover restricted to
#     the top 25% of the population, intensifying around the current
#     best solutions.
#
# This is a *minimal* reproduction of the algorithm — it includes the
# 25/75 phase split, PBX-α with α=0.5, BLX-α with α=0.5, tournament
# selection of size 2, and elitism. It does NOT include the more
# elaborate niching tweaks from the original paper, which are widely
# considered "implementation choices" rather than core to GL-25.

def _run_gl25(obj_func, dim, lb, ub, pop_size, max_fes,
              err_tol, f_opt, seed) -> RunResult:
    rng = np.random.default_rng(seed)
    instr = _InstrumentedObjective(obj_func, max_fes, err_tol, f_opt)

    lb_a = np.broadcast_to(lb, dim).astype(float)
    ub_a = np.broadcast_to(ub, dim).astype(float)

    pop = lb_a + rng.random((pop_size, dim)) * (ub_a - lb_a)
    fit = np.full(pop_size, np.inf)
    try:
        for i in range(pop_size):
            fit[i] = instr(pop[i])

        switch_at = int(0.25 * max_fes)   # 25% of budget for global phase

        while instr.fes_used < max_fes:
            # Phase selection: global vs local
            global_phase = instr.fes_used < switch_at

            if global_phase:
                # Tournament-2 pick from full population
                i1, i2 = rng.integers(0, pop_size, size=2)
                p1 = i1 if fit[i1] < fit[i2] else i2
                # Female is from the alpha males (top 25%)
                alphas = np.argsort(fit)[: max(1, pop_size // 4)]
                p2 = int(rng.choice(alphas))
                # PBX-α (α=0.5)
                alpha = 0.5
                center = pop[p1]
                halfwidth = alpha * np.abs(pop[p2] - pop[p1])
                child = center + (rng.random(dim) * 2 - 1) * halfwidth
            else:
                # Local phase: BLX-α restricted to top 25%
                alphas = np.argsort(fit)[: max(2, pop_size // 4)]
                p1, p2 = rng.choice(alphas, size=2, replace=False)
                alpha = 0.5
                lo = np.minimum(pop[p1], pop[p2])
                hi = np.maximum(pop[p1], pop[p2])
                width = hi - lo
                child = lo - alpha * width + rng.random(dim) * (1 + 2 * alpha) * width

            # Gaussian mutation with small probability
            mask = rng.random(dim) < (1.0 / dim)
            child = np.where(mask,
                             child + 0.1 * (ub_a - lb_a) * rng.standard_normal(dim),
                             child)
            child = np.clip(child, lb_a, ub_a)

            f_child = instr(child)
            # Replace worst if better (elitist replacement)
            worst = int(np.argmax(fit))
            if f_child < fit[worst]:
                pop[worst] = child
                fit[worst] = f_child
    except _BudgetReached:
        pass

    return _make_result(instr, dim)


# ──────────────────────────────────────────────────────────────────────
#  Registry — uniform runner signature
# ──────────────────────────────────────────────────────────────────────


# Canonical 12-algorithm comparator list (paper §4.1)
COMPARATORS_12 = ["GWO", "HHO", "SMA", "RUN",
                  "ABC", "RCGA", "PSO", "DE",
                  "CLPSO", "GL-25", "SADE", "LSHADE"]


# ──────────────────────────────────────────────────────────────────────
#  CEO and FFOA runners — wrapped in the same FE-instrumented interface
# ──────────────────────────────────────────────────────────────────────


def _run_ceo(obj_func, dim, lb, ub, pop_size, max_fes, err_tol, f_opt, seed,
             n_chaos: int = 1) -> RunResult:
    """Run CEO with the official Dong-et-al protocol.

    Per §4.1 of the paper, n_chaos defaults to 1 except for Ackley (5)
    and Rastrigin (20) — caller must pass the right value via the
    `n_chaos` kwarg threaded through the runner factory below.
    """
    instr = _InstrumentedObjective(obj_func, max_fes, err_tol, f_opt)
    stopped_normally = False
    try:
        opt = CEO(obj_func=instr, dim=dim, lb=lb, ub=ub,
                  pop_size=pop_size, max_fes=max_fes,
                  n_chaos=n_chaos, err_tol=err_tol, f_opt=f_opt,
                  seed=seed, verbose=False)
        opt.optimize()
        stopped_normally = True
    except _BudgetReached:
        pass
    termination = None
    if (
        stopped_normally
        and instr.fes_to_target is None
        and instr.fes_used < max_fes
    ):
        termination = "stagnation"
    return _make_result(instr, dim, termination)


def _run_ffoa(obj_func, dim, lb, ub, pop_size, max_fes, err_tol, f_opt, seed
              ) -> RunResult:
    instr = _InstrumentedObjective(obj_func, max_fes, err_tol, f_opt)
    pop_min = max(4, pop_size // 5)
    max_iter = max(1, int(np.ceil(max_fes / pop_min)) + 1)
    try:
        opt = FFOA(obj_func=instr, dim=dim, lb=lb, ub=ub,
                   pop_size=pop_size, max_iter=max_iter,
                   max_fes=max_fes, pop_min=pop_min, seed=seed)
        opt.optimize()
    except (_BudgetReached, EvaluationBudgetReached):
        pass
    return _make_result(instr, dim)


# Per-function CEO n_chaos overrides (paper §4.1)
CEO_N_CHAOS_OVERRIDES = {"Ackley": 5, "Rastrigin": 20}


def make_runner(algo_name: str, fname: str | None = None) -> Callable:
    """Return a function with the signature
        runner(obj_func, dim, lb, ub, pop_size, max_fes, err_tol, f_opt, seed)
    that runs the named algorithm.

    For CEO, `fname` selects the per-function `n_chaos` (paper §4.1):
    Ackley → 5, Rastrigin → 20, everything else → 1.
    """
    name = algo_name.upper()
    if name == "CEO":
        nc = CEO_N_CHAOS_OVERRIDES.get(fname or "", 1)
        return lambda *a, **k: _run_ceo(*a, n_chaos=nc, **k)
    if name == "FFOA":
        return _run_ffoa
    if name == "GWO":
        return lambda *a, **k: _run_mealpy(OriginalGWO, *a, **k)
    if name == "HHO":
        return lambda *a, **k: _run_mealpy(OriginalHHO, *a, **k)
    if name == "SMA":
        return lambda *a, **k: _run_mealpy(OriginalSMA, *a, **k)
    if name == "RUN":
        return lambda *a, **k: _run_mealpy(OriginalRUN, *a, **k)
    if name == "ABC":
        return lambda *a, **k: _run_mealpy(OriginalABC, *a, **k)
    if name == "RCGA":
        # BaseGA is Mealpy's implemented real-coded GA. OriginalGA in
        # Mealpy 3.0.3 has no evolve() method and only evaluates its
        # initial population.
        return lambda *a, **k: _run_mealpy(
            BaseGA, *a, pc=0.9, pm=0.05, selection="tournament",
            crossover="arithmetic", mutation="flip",
            mutation_multipoints=True, **k
        )
    if name == "PSO":
        return lambda *a, **k: _run_mealpy(OriginalPSO, *a, **k)
    if name == "DE":
        # Canonical DE/rand/1/bin parameters (Storn & Price 1997):
        # F (weighting factor) = 0.5, CR (crossover) = 0.9. mealpy's
        # default of F=0.1 is starvation-level and unfair to the algorithm.
        return lambda *a, **k: _run_mealpy(OriginalDE, *a, wf=0.5, cr=0.9, **k)
    if name == "CLPSO":
        return lambda *a, **k: _run_mealpy(CL_PSO, *a, **k)
    if name in ("GL25", "GL-25"):
        return lambda *a, **k: _run_gl25(*a, **k)
    if name == "SADE":
        return lambda *a, **k: _run_mealpy(SADE, *a, **k)
    if name == "LSHADE":
        return lambda *a, **k: _run_mealpy(L_SHADE, *a, **k)
    raise ValueError(f"Unknown algorithm: {algo_name}")


# All algorithms for the full sweep (14 total = 12 paper comparators + CEO + FFOA)
ALL_ALGOS = ["CEO", "FFOA"] + COMPARATORS_12


if __name__ == "__main__":
    """Self-test: run every comparator on Sphere D=10 and verify each returns."""
    import time
    from ceo_benchmarks import _sphere

    print(f"{'Algo':<8} {'f_best':>14} {'fes_to_1e-8':>14} {'time(s)':>9}")
    print("-" * 50)
    for algo in COMPARATORS_12:
        runner = make_runner(algo)
        t0 = time.time()
        r = runner(_sphere, 10, -100, 100, 50, 100_000, 1e-8, 0.0, 0)
        dt = time.time() - t0
        fes_str = f"{r.fes_to_target:>14d}" if r.fes_to_target is not None else f"{'N/A':>14}"
        print(f"{algo:<8} {r.f_best:>14.4e} {fes_str:>14} {dt:>9.2f}")
