"""Validated CEC function factory used by the corrected experiments.

Opfunu's CEC-2017 hybrid F17 creates a one-dimensional elliptic
subcomponent at D=10. Its generic elliptic implementation divides by
``ndim - 1`` and returns NaN for that valid case. The mathematical
one-dimensional limit is simply ``x[0]**2``.
"""

from __future__ import annotations

import math

import numpy as np
import opfunu
from opfunu.utils import operator


def elliptic_finite(x) -> float:
    values = np.asarray(x, dtype=float).ravel()
    if values.size == 0:
        return 0.0
    if values.size == 1:
        return float(values[0] ** 2)
    indices = np.arange(values.size, dtype=float)
    weights = 10.0 ** (6.0 * indices / (values.size - 1))
    return float(np.sum(weights * values ** 2))


# CEC classes resolve this function through the shared operator module.
operator.elliptic_func = elliptic_finite


def make_cec_function(suite: str, function_id: int, dim: int):
    class_name = f"F{int(function_id)}{suite}"
    if not hasattr(opfunu.cec_based, class_name):
        raise ValueError(f"Unsupported CEC function: {class_name}")
    function = getattr(opfunu.cec_based, class_name)(ndim=int(dim))
    probes = (
        np.asarray(function.x_global, dtype=float),
        np.zeros(dim, dtype=float),
    )
    for probe in probes:
        value = float(function.evaluate(probe))
        if not math.isfinite(value):
            raise ValueError(
                f"{class_name} D={dim} returned a non-finite validation value"
            )
    return function


def validate_suite(suite: str, dims: list[int],
                   function_ids: list[int]) -> list[dict]:
    records = []
    for dim in dims:
        for function_id in function_ids:
            function = make_cec_function(suite, function_id, dim)
            rng = np.random.default_rng(10_000 + dim * 100 + function_id)
            point = np.asarray(function.lb) + rng.random(dim) * (
                np.asarray(function.ub) - np.asarray(function.lb)
            )
            values = [
                float(function.evaluate(function.x_global)),
                float(function.evaluate(point)),
            ]
            records.append({
                "suite": suite,
                "dimension": dim,
                "function_id": function_id,
                "finite": all(math.isfinite(value) for value in values),
                "values": values,
            })
    return records

