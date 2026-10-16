"""
CEO paper Table 1 — 15 benchmark functions, with shifted variants.

Verbatim reproduction of Table 1 in Dong et al. (2025), "Chaotic evolution
optimization: A novel metaheuristic algorithm inspired by chaotic
dynamics" (Chaos, Solitons & Fractals 192:116049).

Each function exposes:
    fn        — callable f(x) -> float
    name      — short label used in plots/tables
    dim_free  — True if the function accepts any dim ≥ 2
    lb, ub    — scalar bounds (Table 1)
    x_opt     — analytic optimum (broadcastable to any dim)
    f_opt     — function value at the optimum (always 0 in Table 1)
    needs_x_opt_shifted_inside : bool
        If True, the shift vector for the shifted variant is sampled
        inside the box [lb+pad, ub-pad] so the shifted optimum is
        always feasible. The pad is 0.5 × half-range (matches the
        common shifted-benchmark convention for non-zero optima).

Note on Rosenbrock: the paper writes it with two slightly different
forms in Table 1 (rows "Rosenbrock" and "Schwefel 2.4" are both
quadratic-product style). We use the standard, well-defined Rosenbrock
of Storn & Price; the "Schwefel 2.4" entry in Table 1 reduces to the
same valley landscape under [0,10] bounds.

A "shifted" wrapper shifts the input by a random vector o sampled with
the run's RNG seed so a single benchmark cell has a fixed, reproducible
shift across all algorithms on that seed. The optimum value remains 0
(or whatever the function returns at its analytic optimum).
"""

from __future__ import annotations

import hashlib
import math
import numpy as np
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Callable, Tuple


# ──────────────────────────────────────────────────────────────────────
#  The 15 base functions (Table 1)
# ──────────────────────────────────────────────────────────────────────


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))


def _schwefel_2_22(x: np.ndarray) -> float:
    a = np.abs(x)
    return float(np.sum(a) + np.prod(a))


def _schwefel_1_2(x: np.ndarray) -> float:
    cs = np.cumsum(x)
    return float(np.sum(cs * cs))


def _rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2))


def _schwefel_2_4(x: np.ndarray) -> float:
    # Table 1: sum_i (xi - 1)^2 + (x1 - xi^2)^2. Optimum at x = 1.
    return float(np.sum((x - 1.0) ** 2 + (x[0] - x ** 2) ** 2))


def _elliptic(x: np.ndarray) -> float:
    n = len(x)
    if n == 1:
        return float(x[0] ** 2)
    weights = 1e6 ** (np.arange(n) / (n - 1))
    return float(np.sum(weights * x * x))


def _tablet(x: np.ndarray) -> float:
    # Paper Table 1: 1e6 * x1^2 + sum_{i=2..n} x_i^6.
    return float(1e6 * x[0] ** 2 + np.sum(x[1:] ** 6))


def _zakharov(x: np.ndarray) -> float:
    s1 = np.sum(x * x)
    s2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
    return float(s1 + s2 ** 2 + s2 ** 4)


def _u_penalty(x: np.ndarray, a: float, k: float, m: float) -> np.ndarray:
    out = np.zeros_like(x)
    above = x > a
    below = x < -a
    out[above] = k * (x[above] - a) ** m
    out[below] = k * (-x[below] - a) ** m
    return out


def _penalized_1(x: np.ndarray) -> float:
    n = len(x)
    y = 1.0 + (x + 1.0) / 4.0
    term = 10.0 * np.sin(np.pi * y[0]) ** 2
    term += np.sum((y[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * y[1:]) ** 2))
    term += (y[-1] - 1.0) ** 2
    val = (np.pi / n) * term + float(np.sum(_u_penalty(x, 10.0, 100.0, 4.0)))
    return float(val)


def _penalized_2(x: np.ndarray) -> float:
    n = len(x)
    term = np.sin(3.0 * np.pi * x[0]) ** 2
    term += np.sum((x[:-1] - 1.0) ** 2 * (1.0 + np.sin(3.0 * np.pi * x[1:]) ** 2))
    term += (x[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * x[-1]) ** 2)
    val = 0.1 * term + float(np.sum(_u_penalty(x, 5.0, 100.0, 4.0)))
    return float(val)


def _ackley(x: np.ndarray) -> float:
    n = len(x)
    s1 = np.sum(x * x) / n
    s2 = np.sum(np.cos(2.0 * np.pi * x)) / n
    return float(20.0 + math.e - 20.0 * np.exp(-0.2 * np.sqrt(s1)) - np.exp(s2))


def _griewank(x: np.ndarray) -> float:
    n = len(x)
    s1 = np.sum(x * x) / 4000.0
    s2 = float(reduce(mul, (math.cos(xi / math.sqrt(i + 1)) for i, xi in enumerate(x))))
    return float(s1 - s2 + 1.0)


def _rastrigin(x: np.ndarray) -> float:
    return float(np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x) + 10.0))


def _levy_montalvo_1(x: np.ndarray) -> float:
    n = len(x)
    y = 1.0 + 0.25 * (x + 1.0)
    term = 10.0 * np.sin(np.pi * y[0]) ** 2
    term += np.sum((y[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * y[1:]) ** 2))
    term += (y[-1] - 1.0) ** 2
    return float((np.pi / n) * term)


def _levy_montalvo_2(x: np.ndarray) -> float:
    term = np.sin(3.0 * np.pi * x[0]) ** 2
    term += np.sum((x[:-1] - 1.0) ** 2 * (1.0 + np.sin(3.0 * np.pi * x[1:]) ** 2))
    term += (x[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * x[-1]) ** 2)
    return float(0.1 * term)


# ──────────────────────────────────────────────────────────────────────
#  Public registry
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Benchmark:
    name: str
    fn: Callable[[np.ndarray], float]
    lb: float
    ub: float
    x_opt_value: float   # scalar value of every component of x_opt
    f_opt: float = 0.0

    def x_opt(self, dim: int) -> np.ndarray:
        return np.full(dim, self.x_opt_value, dtype=float)

    def bounds(self, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        return (np.full(dim, self.lb), np.full(dim, self.ub))


# Ordering matches Table 1, top-to-bottom.
BENCHMARKS: list[Benchmark] = [
    Benchmark("Sphere",            _sphere,        -100,  100,  0.0),
    Benchmark("Schwefel 2.22",     _schwefel_2_22, -10,   10,   0.0),
    Benchmark("Schwefel 1.2",      _schwefel_1_2,  -100,  100,  0.0),
    Benchmark("Rosenbrock",        _rosenbrock,    -30,   30,   1.0),
    Benchmark("Schwefel 2.4",      _schwefel_2_4,  0.0,   10.0, 1.0),
    Benchmark("Elliptic",          _elliptic,      -100,  100,  0.0),
    Benchmark("Tablet",            _tablet,        -100,  100,  0.0),
    Benchmark("Zakharov",          _zakharov,      -5.0,  10.0, 0.0),
    Benchmark("Penalized 1",       _penalized_1,   -50.0, 50.0, -1.0),
    Benchmark("Penalized 2",       _penalized_2,   -50.0, 50.0, 1.0),
    Benchmark("Ackley",            _ackley,        -32.0, 32.0, 0.0),
    Benchmark("Griewank",          _griewank,      -600.0, 600.0, 0.0),
    Benchmark("Rastrigin",         _rastrigin,     -5.12, 5.12, 0.0),
    Benchmark("Levy & Montalvo 1", _levy_montalvo_1, -10.0, 10.0, -1.0),
    Benchmark("Levy & Montalvo 2", _levy_montalvo_2, -5.0, 5.0,   1.0),
]

BENCHMARK_BY_NAME: dict[str, Benchmark] = {b.name: b for b in BENCHMARKS}


# ──────────────────────────────────────────────────────────────────────
#  Shifted-variant factory
# ──────────────────────────────────────────────────────────────────────


SHIFT_MASTER_SEED = 2025


def shift_fingerprint(b: Benchmark, dim: int,
                      master_seed: int = SHIFT_MASTER_SEED) -> str:
    payload = f"ceo-shift-v1|{master_seed}|{b.name}|D{dim}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_shifted(
    b: Benchmark,
    dim: int,
    master_seed: int = SHIFT_MASTER_SEED,
) -> Tuple[Callable, np.ndarray, np.ndarray, float, np.ndarray]:
    """Create a shifted version of benchmark `b`.

    The shift vector `o` is sampled deterministically from a local RNG
    seeded from SHA-256 of `(master_seed, b.name, dim)` so all
    algorithms and optimization seeds in a function-dimension cell see
    the same instance, including across Python processes. The shift is
    restricted to
    a sub-box that keeps the function's analytic optimum strictly
    inside [lb, ub] under the substitution x ← x - o + x_opt.

    Returns (g, lb, ub, f_opt, o) where g(x) = b.fn(x - o + x_opt) and
    g has its optimum at o (with the same f_opt as the unshifted form).
    """
    lb, ub = b.bounds(dim)
    x_opt = b.x_opt(dim)
    span = (ub - lb)
    # Keep optimum at least 20% away from each wall so reflective bound
    # repair has somewhere to bounce; mirrors the convention used in
    # most shifted CEC benchmarks.
    pad = 0.2 * span
    digest = shift_fingerprint(b, dim, master_seed)
    rng_seed = int.from_bytes(bytes.fromhex(digest[:16]), "big", signed=False)
    rng = np.random.default_rng(rng_seed)
    o = lb + pad + rng.random(dim) * (span - 2 * pad)

    def g(x: np.ndarray) -> float:
        return b.fn(x - o + x_opt)

    return g, lb, ub, b.f_opt, o


def make_unshifted(b: Benchmark, dim: int) -> Tuple[Callable, np.ndarray, np.ndarray, float, np.ndarray]:
    """Unshifted variant: returns the function and its analytic optimum."""
    lb, ub = b.bounds(dim)
    return b.fn, lb, ub, b.f_opt, b.x_opt(dim)


# ──────────────────────────────────────────────────────────────────────
#  Sanity check (runs as a script)
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Verify every Table-1 function evaluates to ~f_opt at its claimed optimum."""
    print(f"{'Function':<22} {'dim':>4} {'f(x_opt)':>16} {'f_opt':>9}")
    print("-" * 56)
    ok = True
    for b in BENCHMARKS:
        for dim in (2, 5, 10, 20, 30):
            x = b.x_opt(dim)
            val = b.fn(x)
            err = val - b.f_opt
            tag = "ok " if abs(err) < 1e-9 else "FAIL"
            if abs(err) >= 1e-9:
                ok = False
            print(f"{b.name:<22} {dim:>4} {val:>16.6e} {b.f_opt:>9.2f}  {tag}")
    print("-" * 56)
    print("ALL OK" if ok else "SOME FAILED")
