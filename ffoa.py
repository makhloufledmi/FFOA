"""Canonical Fennec Fox Optimization Algorithm (FFOA).

This module is the single executable FFOA implementation used by the
experiments. It exposes the canonical FFOA profile as a compact
fennec-inspired adaptive swarm optimizer, not as a literal biological
simulation.

The three fennec-grounded search mechanisms are:
    * acoustic pounce search toward promising prey-location hypotheses,
    * successful-hunt memory for F/CR adaptation, and
    * remembered food-cache reuse through a bounded archive.

Burrow search, stagnation recovery, diversity dispersal, and population
reduction are explicitly identified as generic numerical support operators.
The method does not claim cooperative hunting or elder imitation: fennecs
are social but normally hunt individually.

License: MIT
"""

import numpy as np
from typing import Callable, Tuple, List


class EvaluationBudgetReached(Exception):
    """Raised before an objective call would exceed ``max_fes``."""


class _FFOACore:
    """
    Internal performance core for the canonical FFOA implementation.

    Parameters
    ----------
    obj_func : callable
        Objective function f(x) -> float to MINIMIZE.
    dim : int
        Problem dimension.
    lb, ub : float or np.ndarray
        Search-space bounds.
    pop_size : int
        Initial pack size (default 30).
    max_iter : int
        Season length in iterations (default 500).
    pop_min : int or None
        Minimum pack size after numerical population reduction
        (default max(4, pop_size//5)).
    cache_size : int
        Food-cache capacity (default 30).
    stagnation_cache : int
        Per-fox failed hunts before returning to cache (default 20).
    stagnation_escape : int
        Per-fox failed hunts before predator-escape (default 50).
    mu : float
        Metabolic-envelope decay exponent (default 2.0).
    gamma : float
        Burrow-digging noise amplitude (default 0.1).
    H : int
        Episodic prey-patch memory depth; number of successful hunting
        episodes retained in success-history memory (default 10).
    p_max : float
        Initial prey-hypothesis pool fraction (default 0.20).
    p_min : float
        Final prey-hypothesis pool fraction (default 0.05).
    diversity_thr : float
        Relative diversity threshold triggering prey-migration dispersal
        (default 0.01).
    restart_patience : int
        Global-best stagnation count before dispersal fires (default 30).
    restart_frac : float
        Fraction of worst foxes dispersed when prey has migrated
        (default 0.30).
    seed : int or None
        Random seed.
    verbose : bool
        Print season progress every 50 iterations.
    """

    def __init__(self, obj_func: Callable, dim: int, lb, ub,
                 pop_size: int = 30,
                 max_iter: int = 500,
                 max_fes: int = None,
                 pop_min: int = None,
                 cache_size: int = 30,
                 stagnation_cache: int = 20,
                 stagnation_escape: int = 50,
                 mu: float = 2.0,
                 gamma: float = 0.1,
                 H: int = 10,
                 p_max: float = 0.20,
                 p_min: float = 0.05,
                 diversity_thr: float = 0.01,
                 restart_patience: int = 30,
                 restart_frac: float = 0.30,
                 seed: int = None,
                 verbose: bool = False):

        self.f = obj_func
        self.dim = dim
        self.lb = np.full(dim, lb) if np.isscalar(lb) else np.asarray(lb, float)
        self.ub = np.full(dim, ub) if np.isscalar(ub) else np.asarray(ub, float)
        self.N_init = pop_size
        self.N = pop_size
        self.N_min = (pop_min if pop_min is not None
                      else (max(4, pop_size // 5) if pop_size > 10 else pop_size))
        self.T = max_iter
        self.max_fes = int(max_fes) if max_fes is not None else None
        if self.max_fes is not None and self.max_fes < self.N_init:
            raise ValueError("max_fes must be at least pop_size")
        self.fes_used = 0
        self.cache_size = cache_size
        self.k_cache = stagnation_cache
        self.k_escape = stagnation_escape
        self.mu = mu
        self.gamma = gamma
        self.H = H
        self.p_max = p_max
        self.p_min = p_min
        self.diversity_thr = diversity_thr
        self.restart_patience = restart_patience
        self.restart_frac = restart_frac
        self.verbose = verbose

        if seed is not None:
            np.random.seed(seed)

        # ── population arrays (allocated at _initialize) ──────────────
        self.X: np.ndarray = None
        self.fitness: np.ndarray = None
        self.X_best: np.ndarray = None
        self.f_best: float = np.inf
        self.X_burrow: np.ndarray = None
        self.f_burrow: np.ndarray = None
        self.stagnation: np.ndarray = None

        # ── food cache ────────────────────────────────────────────────
        self.cache_X: List[np.ndarray] = []
        self.cache_f: List[float] = []

        # Bounded archive interpreted as remembered food-cache locations.
        self._archive: List[np.ndarray] = []
        self._archive_cap: int = self.N_init

        # Episodic prey-patch memory for F/CR adaptation.
        self._F_memory  = np.full(H, 0.5)
        self._CR_memory = np.full(H, 0.5)
        self._F_mem_idx: int = 0
        self._CR_mem_idx: int = 0
        self._F_success:  List[float] = []
        self._F_delta:    List[float] = []
        self._CR_success: List[float] = []
        self._CR_delta:   List[float] = []
        self._F_used:  List = []
        self._CR_used: List = []

        # Prey-migration dispersal tracker.
        self._no_improve_count: int = 0
        self._last_best: float = np.inf

        # ── observables ───────────────────────────────────────────────
        self.convergence_curve:   List[float] = []
        self.diversity_curve:     List[float] = []
        self.search_history_x1:   List[float] = []
        self.search_history_x2:   List[float] = []
        self.trajectory_first:    List[float] = []
        self.avg_fitness_history:  List[float] = []
        self.escaping_energy:      List[float] = []

    # ══════════════════════════════════════════════════════════════════
    #  Season schedule helpers
    # ══════════════════════════════════════════════════════════════════

    def _progress(self, t: int) -> float:
        if self.max_fes is not None:
            return min(1.0, self.fes_used / self.max_fes)
        return min(1.0, t / self.T)

    def _evaluate(self, x: np.ndarray) -> float:
        if self.max_fes is not None and self.fes_used >= self.max_fes:
            raise EvaluationBudgetReached
        value = float(self.f(x))
        self.fes_used += 1
        return value

    def _envelope(self, t: int) -> float:
        """Metabolic-decay envelope: energy available declines with season."""
        return (1.0 - self._progress(t)) ** self.mu

    def _acoustic_activity(self, t: int) -> float:
        """Seasonal acoustic-primary foraging schedule.

        Fennec foxes spend most active foraging time in binaural
        acoustic scanning mode (Gauthier-Clerc et al. 1998).  Burrowing
        follows acoustic confirmation; it is not the primary behaviour.
        A_t is the probability of an acoustic pounce proposal
        this iteration.  It ranges from 1.0 early in the season (peak
        acoustic activity) down to 0.85 at season's end, when deeper-
        buried prey require more direct digging.
        """
        env = self._envelope(t)
        return 0.85 + 0.15 * env

    def _prey_pool_fraction(self, t: int) -> float:
        """FE-normalized narrowing of the prey-location hypothesis pool."""
        return self.p_max - (self.p_max - self.p_min) * self._progress(t)

    def _population_target_size(self, t: int) -> int:
        frac = self._progress(t)
        n = int(round(self.N_init - frac * (self.N_init - self.N_min)))
        return max(self.N_min, n)

    # ══════════════════════════════════════════════════════════════════
    #  Initialisation
    # ══════════════════════════════════════════════════════════════════

    def _initialize(self):
        self.X = self.lb + np.random.rand(self.N, self.dim) * (self.ub - self.lb)
        self.fitness = np.array([self._evaluate(x) for x in self.X])
        self.stagnation = np.zeros(self.N, dtype=int)
        self.X_burrow = self.X.copy()
        self.f_burrow = self.fitness.copy()
        self._update_leader()
        for x, fv in zip(self.X, self.fitness):
            self._cache_insert(x, fv)
        self._last_best = self.f_best

    def _update_leader(self):
        idx = int(np.argmin(self.fitness))
        if self.fitness[idx] < self.f_best:
            self.f_best = float(self.fitness[idx])
            self.X_best = self.X[idx].copy()

    # ══════════════════════════════════════════════════════════════════
    #  Food-cache memory and displaced-position archive
    # ══════════════════════════════════════════════════════════════════

    def _cache_insert(self, x: np.ndarray, fv: float):
        if len(self.cache_X) < self.cache_size:
            self.cache_X.append(x.copy())
            self.cache_f.append(float(fv))
        else:
            worst = int(np.argmax(self.cache_f))
            if fv < self.cache_f[worst]:
                self.cache_X[worst] = x.copy()
                self.cache_f[worst] = float(fv)

    def _archive_insert(self, x: np.ndarray):
        if len(self._archive) >= self._archive_cap:
            j = np.random.randint(len(self._archive))
            self._archive[j] = x.copy()
        else:
            self._archive.append(x.copy())

    def _archive_shrink(self, cap: int):
        self._archive_cap = max(1, cap)
        while len(self._archive) > self._archive_cap:
            self._archive.pop(np.random.randint(len(self._archive)))

    # ══════════════════════════════════════════════════════════════════
    #  Operator 2 - Individual acoustic pounce search (core operator)
    # ══════════════════════════════════════════════════════════════════

    def _acoustic_pounce(self, i: int, t: int) -> np.ndarray:
        """Individual hunt toward a promising prey-location hypothesis.

        Population members represent concurrent search trajectories. A
        high-quality location is sampled as the current prey hypothesis;
        this is information sharing between numerical trajectories, not a
        claim that fennecs hunt cooperatively or imitate elite elders.
        Movement scale F and coordinate commitment CR are drawn from the
        successful-hunt memory.
        """
        # F from episodic prey-patch memory.
        mu_F = self._F_memory[np.random.randint(self.H)]
        F = -1.0
        while F <= 0.0:
            F = mu_F + 0.1 * np.random.standard_cauchy()
            if F > 1.0:
                F = 1.0
                break

        # CR from episodic prey-patch memory.
        mu_CR = self._CR_memory[np.random.randint(self.H)]
        CR = float(np.clip(np.random.normal(mu_CR, 0.1), 0.0, 1.0))

        # The set of plausible prey locations narrows with FE progress.
        p = self._prey_pool_fraction(t)
        top_k = max(1, int(np.ceil(p * self.N)))
        order = np.argsort(self.fitness)[:top_k]
        x_prey = self.X[int(np.random.choice(order))]

        # ── r1 from live pack ─────────────────────────────────────────
        others = [j for j in range(self.N) if j != i]
        r1 = int(np.random.choice(others))

        # r2 comes from the live population or remembered food-cache archive.
        pool = list(range(self.N)) + [self.N + k for k in range(len(self._archive))]
        pool = [c for c in pool if c != i]
        if pool:
            r2c = int(np.random.choice(pool))
            x_r2 = self.X[r2c] if r2c < self.N else self._archive[r2c - self.N]
        else:
            x_r2 = self.X[r1]

        self._F_used[i]  = F
        self._CR_used[i] = CR

        # Acoustic prey pursuit plus food-cache-assisted variation.
        v = self.X[i] + F * (x_prey - self.X[i]) + F * (self.X[r1] - x_r2)

        # Selective pounce commitment in a subset of coordinates.
        u = self.X[i].copy()
        mask = np.random.rand(self.dim) < CR
        mask[np.random.randint(self.dim)] = True
        u[mask] = v[mask]
        return u

    # ══════════════════════════════════════════════════════════════════
    #  Operator 3 — Subspace burrow digging  (secondary, ≤15 % of FEs)
    # ══════════════════════════════════════════════════════════════════

    def _burrow_digging(self, i: int, env: float) -> np.ndarray:
        """Directed tunnel extension toward the alpha's burrow.

        Called when the seasonal acoustic schedule selects direct
        digging rather than acoustic learning. A fox digs toward the
        best-known safe spot in a random subspace, adding shrinking
        Gaussian tunnel-widening noise.
        """
        X_new = self.X[i].copy()
        n_perturb = max(1, int(round(self.dim * env)))
        dims = np.random.choice(self.dim, n_perturb, replace=False)
        direction = self.X_best[dims] - self.X[i][dims]
        X_new[dims] += env * 0.5 * direction
        X_new[dims] += (self.gamma
                        * (self.ub[dims] - self.lb[dims])
                        * env
                        * np.random.randn(n_perturb))
        return X_new

    # ══════════════════════════════════════════════════════════════════
    #  Operator 4 — Cache-and-exhume memory
    # ══════════════════════════════════════════════════════════════════

    def _cache_exhume(self, i: int, t: int) -> np.ndarray:
        """Return to a cached food store after prolonged failed hunts.

        Triggered only after satiation patience is exhausted. The fox
        casts about for fresh prey near the cached location; search
        radius shrinks geometrically as season energy declines.
        """
        if not self.cache_X:
            return self.X[i].copy()
        order = np.argsort(self.cache_f)
        ranks = np.arange(1, len(order) + 1, dtype=float)
        weights = 1.0 / ranks
        weights /= weights.sum()
        pick = np.random.choice(order, p=weights)
        x_cache = self.cache_X[pick]
        d = np.linalg.norm(self.X[i] - x_cache)
        env = self._envelope(t)
        r_local = 0.3 * d * env + 1e-12
        return x_cache + r_local * np.random.randn(self.dim)

    # ══════════════════════════════════════════════════════════════════
    #  Operator 5 — Predator threat repulsion
    # ══════════════════════════════════════════════════════════════════

    def _predator_escape(self, i: int) -> np.ndarray:
        """Flee worst predators toward personal burrow after extreme stagnation.

        Triggered only after k_escape failed steps, reflecting biological
        patience before a panic response.
        """
        env = self._envelope(len(self.convergence_curve) + 1)
        n_worst = max(1, int(np.ceil(0.1 * self.N)))
        X_pred = np.mean(self.X[np.argsort(self.fitness)[-n_worst:]], axis=0)
        flee_vec  = self.X[i] - X_pred
        flee_dist = np.linalg.norm(flee_vec) + 1e-30
        flee_dir  = flee_vec / flee_dist
        safe_vec  = self.X_burrow[i] - self.X[i]
        safe_dist = np.linalg.norm(safe_vec) + 1e-30
        safe_dir  = safe_vec / safe_dist
        phi, psi = np.random.rand(), np.random.rand()
        return self.X[i] + env * (phi * flee_dist * flee_dir
                                  + psi * safe_dist * safe_dir)

    # ══════════════════════════════════════════════════════════════════
    #  Population management
    # ══════════════════════════════════════════════════════════════════

    def _shrink_population(self, target_N: int):
        """Generic linear population-size reduction."""
        if target_N >= self.N:
            return
        order = np.argsort(self.fitness)[:target_N]
        removed = set(range(self.N)) - set(order.tolist())
        for j in removed:
            self._archive_insert(self.X[j])
        self.X        = self.X[order]
        self.fitness  = self.fitness[order]
        self.X_burrow = self.X_burrow[order]
        self.f_burrow = self.f_burrow[order]
        self.stagnation = self.stagnation[order]
        self.N = target_N
        self._archive_shrink(target_N)

    def _prey_migration_dispersal(self, t: int):
        """Prey-migration dispersal.

        When pack diversity collapses and no improvement has been found
        for restart_patience iterations, the prey population has likely
        migrated.  The worst restart_frac foxes (least committed to the
        depleted territory) are scattered uniformly across the full
        search space to locate the new prey concentration.

        Dispersed foxes are evaluated immediately and inserted into the
        active pack so migration remains visible in the same budget
        accounting as other position changes.
        """
        n_scatter = max(1, int(round(self.restart_frac * self.N)))
        order = np.argsort(self.fitness)
        scatter_idx = order[-n_scatter:]   # worst n_scatter individuals
        for j in scatter_idx:
            self._archive_insert(self.X[j])
            X_new = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)
            X_new = np.clip(X_new, self.lb, self.ub)
            fv = self._evaluate(X_new)
            self.X[j]        = X_new
            self.fitness[j]  = fv
            self.X_burrow[j] = X_new.copy()
            self.f_burrow[j] = fv
            self.stagnation[j] = 0
        self._no_improve_count = 0

    # ══════════════════════════════════════════════════════════════════
    #  Main optimisation loop
    # ══════════════════════════════════════════════════════════════════

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        self._initialize()

        if self.dim >= 2:
            self.search_history_x1.extend(self.X[:, 0].tolist())
            self.search_history_x2.extend(self.X[:, 1].tolist())
        self.trajectory_first.append(self.X[0, 0])
        self.avg_fitness_history.append(float(np.mean(self.fitness)))
        self.escaping_energy.append(1.0)

        diag = float(np.linalg.norm(self.ub - self.lb)) + 1e-30

        for t in range(1, self.T + 1):

            # Generic population-size reduction; not a biological claim.
            self._shrink_population(self._population_target_size(t))

            # Season-schedule quantities
            A_t = self._acoustic_activity(t)
            env = self._envelope(t)

            # Clear per-iteration success buffers
            self._F_success.clear();  self._F_delta.clear()
            self._CR_success.clear(); self._CR_delta.clear()
            self._F_used  = [None] * self.N
            self._CR_used = [None] * self.N

            # ── per-fox foraging step ─────────────────────────────────
            for i in range(self.N):
                if self.stagnation[i] > self.k_escape:
                    # Predator escape after prolonged failed hunts.
                    X_new = self._predator_escape(i)
                    self.stagnation[i] = 0

                elif self.stagnation[i] > self.k_cache:
                    # Cache return after moderate failed hunts.
                    X_new = self._cache_exhume(i, t)

                else:
                    # Acoustic hunt is the primary seasonal behaviour.
                    if np.random.rand() < A_t:
                        X_new = self._acoustic_pounce(i, t)
                    else:
                        X_new = self._burrow_digging(i, env)

                X_new = np.clip(X_new, self.lb, self.ub)
                f_new = self._evaluate(X_new)

                if f_new < self.fitness[i]:
                    delta = self.fitness[i] - f_new
                    if self._F_used[i] is not None:
                        self._F_success.append(self._F_used[i])
                        self._F_delta.append(delta)
                    if self._CR_used[i] is not None:
                        self._CR_success.append(self._CR_used[i])
                        self._CR_delta.append(delta)
                    self._archive_insert(self.X[i])
                    self.X[i]       = X_new
                    self.fitness[i] = f_new
                    self.stagnation[i] = 0
                    if f_new < self.f_burrow[i]:
                        self.X_burrow[i] = X_new.copy()
                        self.f_burrow[i] = f_new
                    self._cache_insert(X_new, f_new)
                else:
                    self.stagnation[i] += 1

            # Update episodic prey-patch memory.
            if self._F_success:
                w  = np.array(self._F_delta, float)
                w  = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
                Fs = np.array(self._F_success, float)
                lehmer = float(np.dot(w, Fs * Fs) / (np.dot(w, Fs) + 1e-30))
                self._F_memory[self._F_mem_idx % self.H] = lehmer
                self._F_mem_idx += 1

            if self._CR_success:
                w   = np.array(self._CR_delta, float)
                w   = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
                CRs = np.array(self._CR_success, float)
                self._CR_memory[self._CR_mem_idx % self.H] = float(np.dot(w, CRs))
                self._CR_mem_idx += 1

            self._update_leader()
            self.convergence_curve.append(self.f_best)
            self.diversity_curve.append(float(np.mean(np.std(self.X, axis=0))))

            # Prey-migration dispersal check.
            if self.f_best < self._last_best:
                self._last_best        = self.f_best
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1

            rel_diversity = np.mean(np.std(self.X, axis=0)) / diag
            if (self._no_improve_count >= self.restart_patience
                    and rel_diversity < self.diversity_thr
                    and self.N >= 4):
                self._prey_migration_dispersal(t)
                self._update_leader()
            # ─────────────────────────────────────────────────────────

            self.escaping_energy.append(A_t)
            self.trajectory_first.append(self.X[0, 0])
            self.avg_fitness_history.append(float(np.mean(self.fitness)))
            if t % 10 == 0 or t == 1:
                if self.dim >= 2:
                    self.search_history_x1.extend(self.X[:, 0].tolist())
                    self.search_history_x2.extend(self.X[:, 1].tolist())

            if self.verbose and (t % 50 == 0 or t == 1):
                print(f"Iter {t:4d} | N={self.N} | Best={self.f_best:.4e} | "
                      f"Div={self.diversity_curve[-1]:.4f} | "
                      f"NoImprv={self._no_improve_count}")

        return self.X_best, self.f_best, self.convergence_curve


class FFOA(_FFOACore):
    """Canonical FFOA with three primary user controls.

    Population size, exact FE budget, and seed are the intended controls.
    Internal constants remain accepted as legacy keywords solely so that the
    archived experiments and diagnostic sensitivity script are reproducible.
    Defaults exactly match the validated numerical profile.
    """

    def __init__(
        self,
        obj_func: Callable,
        dim: int,
        lb,
        ub,
        pop_size: int = 30,
        max_iter: int = 500,
        max_fes: int | None = None,
        *,
        seed: int | None = None,
        verbose: bool = False,
        **legacy,
    ):
        pop_min = legacy.pop("pop_min", None)
        cache_size = legacy.pop("cache_size", 30)
        prey_memory = legacy.pop("prey_memory", 10)
        prey_pool = legacy.pop("prey_pool", legacy.pop("elder_pool", (0.20, 0.05)))
        p_start, p_end = prey_pool
        patience = legacy.pop("patience", 20)
        escape_patience = legacy.pop("escape_patience", 50)
        migration_patience = legacy.pop("migration_patience", 30)
        migration_fraction = legacy.pop("migration_fraction", 0.30)
        burrow_radius = legacy.pop("burrow_radius", 0.10)
        diversity_trigger = legacy.pop("diversity_trigger", 0.01)
        metabolic_decay = legacy.pop("metabolic_decay", 2.0)

        metabolic_decay = legacy.pop("mu", metabolic_decay)
        prey_memory = legacy.pop("H", prey_memory)
        p_start = legacy.pop("p_max", p_start)
        p_end = legacy.pop("p_min", p_end)
        patience = legacy.pop("stagnation_cache", patience)
        escape_patience = legacy.pop("stagnation_escape", escape_patience)
        migration_patience = legacy.pop("restart_patience", migration_patience)
        migration_fraction = legacy.pop("restart_frac", migration_fraction)
        burrow_radius = legacy.pop("gamma", burrow_radius)
        diversity_trigger = legacy.pop("diversity_thr", diversity_trigger)

        ignored_legacy = {
            "diurnal_period",
            "theta_split",
            "k_fission",
            "k_merge",
            "k_cooldown",
            "scout_fraction",
            "r_recent",
            "sigma_cluster",
            "n_clusters",
            "stagnation_threshold",
            "alpha_sand",
            "late_multiplicative",
            "late_phase_start",
            "pack_size",
            "sigma_base",
            "beta_noise",
        }
        for key in ignored_legacy:
            legacy.pop(key, None)
        if legacy:
            unexpected = ", ".join(sorted(legacy))
            raise TypeError(f"Unexpected FFOA parameter(s): {unexpected}")

        super().__init__(
            obj_func=obj_func,
            dim=dim,
            lb=lb,
            ub=ub,
            pop_size=pop_size,
            max_iter=max_iter,
            max_fes=max_fes,
            pop_min=pop_min,
            cache_size=cache_size,
            H=prey_memory,
            p_max=p_start,
            p_min=p_end,
            mu=metabolic_decay,
            stagnation_cache=patience,
            stagnation_escape=escape_patience,
            restart_patience=migration_patience,
            restart_frac=migration_fraction,
            gamma=burrow_radius,
            diversity_thr=diversity_trigger,
            seed=seed,
            verbose=verbose,
            **legacy,
        )


__all__ = ["EvaluationBudgetReached", "FFOA"]
