"""
Chaotic Evolution Optimization (CEO).

Faithful Python port of the official implementation released by the authors:
  Dong, Zhang, Zhang, Zhou, Jiang (2025).
  "Chaotic evolution optimization: A novel metaheuristic algorithm
  inspired by chaotic dynamics."  Chaos, Solitons and Fractals, 192:116049.
  https://github.com/Running-Wolf1010/CEO

This file follows the official `CEO.py` source line-for-line. Differences
relative to that file:
  - the calling convention is normalised so the same benchmark harness can
    drive CEO and FFOA with one signature;
  - the FE-to-target counter is recorded for the paper's headline metric
    (AvgFEvals to reach error 1e-8, §4.1) — the official code prints to
    stdout instead;
  - the `verbose` flag silences the printout the official code emits
    every 10 iterations.

The algorithmic behaviour (per-iteration population-range remapping, the
E-DM map seeding from the pair itself, reflective boundary repair, per-
side CR sampling, 50-step stagnation guard) is preserved verbatim.

License: MIT (this port). Original algorithm © 2025 Dong et al.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, List, Tuple


class CEO:
    """
    Chaotic Evolution Optimization (port of the official Python release).

    Parameters
    ----------
    obj_func : callable
        Objective f(x) -> float, MINIMISED.
    dim : int
        Problem dimension.
    lb, ub : float | np.ndarray
        Lower / upper bounds (broadcast to dim if scalar).
    pop_size : int = 50
        Population size Np (must be even, ≥ 2).
    max_fes : int | None
        Function-evaluation cap. Default `dim * 10_000` per §4.1 of the
        paper.
    n_chaos : int = 1
        Chaotic sample count N per pair (§4.1: 1 by default, 5 for Ackley,
        20 for Rastrigin; caller decides).
    err_tol : float = 1e-8
        Error tolerance for the headline metric (FEs to reach this error).
        Does NOT terminate the run — the official code only stops on
        MaxFES or the 50-step stagnation guard.
    f_opt : float = 0.0
        Known optimum value, used to compute `error = f_best - f_opt`
        for the FE-to-target reporting.
    stagnation_window : int = 50
        Consecutive outer iterations with |Δf_best| < 1e-8 before the
        run is terminated early. The official code hard-codes 50.
    seed : int | None
        RNG seed.
    verbose : bool

    The remaining (unused) kwargs are absorbed silently so the same
    constructor signature works inside our shared benchmark harness.
    """

    # Reflective bound-repair scratch values (matches the official code)
    _STAGNATION_TOL = 1e-8

    def __init__(self,
                 obj_func: Callable[[np.ndarray], float],
                 dim: int,
                 lb, ub,
                 pop_size: int = 50,
                 max_fes: int | None = None,
                 n_chaos: int = 1,
                 err_tol: float = 1e-8,
                 f_opt: float = 0.0,
                 stagnation_window: int = 50,
                 seed: int | None = None,
                 verbose: bool = False,
                 **_unused):
        if int(pop_size) % 2 != 0:
            raise ValueError("Np must be even (matches the official CEO.py).")
        self.f = obj_func
        self.dim = int(dim)
        self.Varmin = np.full(self.dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, float).reshape(-1)
        self.Varmax = np.full(self.dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, float).reshape(-1)
        self.Np = int(pop_size)
        self.N = max(1, int(n_chaos))
        self.MaxFES = int(max_fes) if max_fes is not None else self.dim * 10_000
        self.err_tol = float(err_tol)
        self.f_opt = float(f_opt)
        self.stagnation_window = int(stagnation_window)
        self.verbose = bool(verbose)

        if seed is not None:
            np.random.seed(int(seed))

        # Chaotic-domain bounds (official lines 25-26)
        self.low_chacos = np.array([-0.5, -0.25])
        self.up_chacos = np.array([0.5, 0.25])

        # Population init (official lines 17-22)
        self.Population = np.random.rand(self.Np, self.dim) * (self.Varmax - self.Varmin) + self.Varmin
        self.fit = self._fitness_batch(self.Population)
        self.index_best = int(np.argmin(self.fit))
        self.fBest = float(self.fit[self.index_best])
        self.Best = self.Population[self.index_best].copy()
        self.FEvals = self.Np
        self.t = 1

        # Reporting buffers
        self.history: List[float] = [self.fBest]
        self.fes_to_target: int | None = None
        self._check_target()

        # Telemetry (kept for compatibility with the FFOA harness)
        self.convergence_curve: List[float] = [self.fBest]
        self.diversity_curve: List[float] = [float(np.mean(np.std(self.Population, axis=0)))]
        self.search_history_x1: List[float] = []
        self.search_history_x2: List[float] = []
        self.trajectory_first: List[float] = [float(self.Population[0, 0])]
        self.avg_fitness_history: List[float] = [float(np.mean(self.fit))]
        self.escaping_energy: List[float] = [1.0]

        # Stagnation counter (initialised lazily in handle_stagnation, but
        # we keep an attribute so it can be inspected from tests)
        self.counter = 0

    # ──────────────────────────────────────────────────────────────────
    #  Fitness helpers (official lines 32-37)
    # ──────────────────────────────────────────────────────────────────

    def _fitness_batch(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        out = np.zeros(n)
        for i in range(n):
            out[i] = float(self.f(X[i]))
        return out

    # ──────────────────────────────────────────────────────────────────
    #  Reflective boundary repair (official lines 40-45)
    # ──────────────────────────────────────────────────────────────────

    def _bound_constraint(self, v: np.ndarray) -> np.ndarray:
        low = np.tile(self.Varmin, (v.shape[0], 1))
        up = np.tile(self.Varmax, (v.shape[0], 1))
        below = v < low
        above = v > up
        v[below] = np.minimum(up[below], 2 * low[below] - v[below])
        v[above] = np.maximum(low[above], 2 * up[above] - v[above])
        return v

    # ──────────────────────────────────────────────────────────────────
    #  E-DM map (official lines 48-57)
    # ──────────────────────────────────────────────────────────────────

    def _EDM(self, x0: np.ndarray, y0: np.ndarray, n_iter: int) -> np.ndarray:
        """Generate N steps of the E-DM map from initial state (x0, y0).

        Returns a (2N, dim) array obtained by stacking the N x-iterates
        on top of the N y-iterates (matches the official `np.vstack`
        return convention so the caller can slice it the same way).
        """
        k = 2.66
        x = np.zeros((n_iter + 1, self.dim))
        y = np.zeros((n_iter + 1, self.dim))
        x[0, :] = x0
        y[0, :] = y0
        for j in range(n_iter):
            x[j + 1, :] = k * (np.exp(-np.cos(np.pi * y[j, :])) - 1.0) * x[j, :]
            y[j + 1, :] = y[j, :] + x[j, :]
        return np.vstack((x[1:, :], y[1:, :]))

    # ──────────────────────────────────────────────────────────────────
    #  Binomial crossover (official lines 60-67)
    # ──────────────────────────────────────────────────────────────────

    def _binomial_crossover(self, p: np.ndarray, v: np.ndarray, CR: float
                            ) -> np.ndarray:
        """Standard DE binomial crossover with j_rand guarantee."""
        n, d = v.shape
        mask = np.random.rand(n, d) < CR
        rand_cols = np.random.randint(0, d, n)
        # Force at least one mutant gene per row (j_rand guarantee)
        mask[np.arange(n), rand_cols] = True
        return mask * v + (~mask) * p

    # ──────────────────────────────────────────────────────────────────
    #  Stagnation termination (official lines 69-77)
    # ──────────────────────────────────────────────────────────────────

    def _handle_stagnation(self, old: float, new: float) -> bool:
        if abs(old - new) < self._STAGNATION_TOL:
            self.counter += 1
            return self.counter > self.stagnation_window
        else:
            self.counter = 0
            return False

    # ──────────────────────────────────────────────────────────────────
    #  Headline-metric helper (FE count at which 1e-8 error is reached)
    # ──────────────────────────────────────────────────────────────────

    def _check_target(self):
        if self.fes_to_target is None and (self.fBest - self.f_opt) <= self.err_tol:
            self.fes_to_target = self.FEvals

    # ──────────────────────────────────────────────────────────────────
    #  Main loop (official lines 79-144)
    # ──────────────────────────────────────────────────────────────────

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        while self.FEvals < self.MaxFES:
            oldfBest = self.fBest
            rand_num = np.random.permutation(self.Np)
            # Per-iteration population range (NOT the problem bounds)
            ub = np.max(self.Population, axis=0)
            lb = np.min(self.Population, axis=0)
            span_dom = ub - lb + np.finfo(float).eps   # avoid div-by-zero
            span_cha = self.up_chacos - self.low_chacos  # length-2 vector

            for i in range(0, self.Np, 2):
                index = rand_num[i:i + 2]
                xy = self.Population[index, :]   # shape (2, dim)

                # Forward mapping (Eq. 4) — per-iteration range
                # xy_dot[k, j] = (xy[k, j] - lb[j]) / span_dom[j]
                #              * (up_chacos[k] - low_chacos[k]) + low_chacos[k]
                xy_dot = ((xy - lb) / span_dom) * span_cha[:, None] + self.low_chacos[:, None]

                # N chaotic candidates from the E-DM map (Eq. 2)
                chaos_total = self._EDM(xy_dot[0, :], xy_dot[1, :], self.N)

                for k in range(2):  # k=0 → x-side, k=1 → y-side
                    xy_chaos = chaos_total[k * self.N:(k + 1) * self.N, :]

                    # Inverse mapping (Eq. 5) — per-iteration range
                    xy_chaos_dot = ((xy_chaos - self.low_chacos[k]) /
                                    (self.up_chacos[k] - self.low_chacos[k])) * (ub - lb) + lb
                    xy_chaos_dot = self._bound_constraint(xy_chaos_dot)

                    # Mutation — Eq. (7) or Eq. (8) with prob 0.5 each
                    a = np.random.rand(self.N, 1)
                    if np.random.rand() < 0.5:
                        xy_hat = xy[k, :] + a * (xy_chaos_dot - xy[k, :])
                    else:
                        xy_hat = self.Best + a * (xy_chaos_dot - xy[k, :])

                    CR = float(np.random.rand())
                    xy_trial = self._binomial_crossover(xy[k, :], xy_hat, CR)
                    xy_trial = self._bound_constraint(xy_trial)

                    fit_trial = self._fitness_batch(xy_trial)
                    self.FEvals += self.N

                    best_trial_idx = int(np.argmin(fit_trial))
                    best_trial_fit = float(fit_trial[best_trial_idx])

                    # Greedy selection (Eq. 10/11)
                    if best_trial_fit < self.fit[index[k]]:
                        self.Population[index[k], :] = xy_trial[best_trial_idx]
                        self.fit[index[k]] = best_trial_fit

                if self.FEvals >= self.MaxFES:
                    break

            # Update global best across the whole population
            self.index_best = int(np.argmin(self.fit))
            self.fBest = float(self.fit[self.index_best])
            self.Best = self.Population[self.index_best].copy()
            self._check_target()

            # Outer-loop bookkeeping (matches official line 134-142)
            if self._handle_stagnation(oldfBest, self.fBest):
                break

            self.t += 1
            self.history.append(self.fBest)
            self.convergence_curve.append(self.fBest)
            self.diversity_curve.append(float(np.mean(np.std(self.Population, axis=0))))
            self.trajectory_first.append(float(self.Population[0, 0]))
            self.avg_fitness_history.append(float(np.mean(self.fit)))
            self.escaping_energy.append(max(0.0, 1.0 - self.FEvals / self.MaxFES))
            if self.t % 10 == 0 and self.dim >= 2:
                self.search_history_x1.extend(self.Population[:, 0].tolist())
                self.search_history_x2.extend(self.Population[:, 1].tolist())
            if self.verbose and self.t % 10 == 0:
                print(f"iter={self.t}  ObjVal={self.fBest:.16f}")

        return self.Best, self.fBest, self.history
