"""
Top-level HELM solver for the IEEE 5-bus OPF reproduction.

This module provides the public solver interface.

The mathematical coefficient recursion is implemented in
recursion.py. This module calls that recursion and packages
the result into a structured HelmSolution object.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problem import HelmProblem
from .recursion import eq14_mismatch, solve_helm_series


@dataclass(frozen=True)
class HelmSolverConfig:
    """
    Numerical settings for the HELM coefficient calculation.

    Defaults follow Algorithm 1 of the paper:
        n_max = 30,  epsilon = varepsilon = 1e-5  (Eq. 14 mismatches)
    """

    max_order: int = 30
    tolerance: float = 1e-5
    min_order: int = 1


@dataclass
class HelmSolution:
    """Result returned by the HELM solver."""

    coefficients: np.ndarray
    inverse_coefficients: np.ndarray
    reactive_power_coefficients: np.ndarray

    voltage: np.ndarray
    voltage_magnitude: np.ndarray
    voltage_angle: np.ndarray

    active_power: np.ndarray
    reactive_power: np.ndarray

    converged: bool
    order: int
    residual: float

    convergence_history: np.ndarray


def evaluate_voltage_series(
    coefficients: np.ndarray,
    z: float = 1.0,
) -> np.ndarray:
    """
    Evaluate the voltage power series at embedding parameter z.

    V(z) = V(0) + V(1)z + V(2)z² + ...

    Parameters
    ----------
    coefficients:
        Array with shape (order + 1, n_bus).

    z:
        Embedding parameter.

    Returns
    -------
    np.ndarray
        Complex bus-voltage vector.
    """

    coefficients = np.asarray(
        coefficients,
        dtype=complex,
    )

    if coefficients.ndim != 2:
        raise ValueError(
            "Voltage coefficients must have shape "
            "(order + 1, n_bus)."
        )

    voltage = np.zeros(
        coefficients.shape[1],
        dtype=complex,
    )

    for order in range(coefficients.shape[0]):
        voltage += coefficients[order] * z**order

    return voltage


def power_flow_residual(
    problem: HelmProblem,
    voltage: np.ndarray,
) -> float:
    """
    Maximum mismatch of the paper's termination conditions (14a)-(14c).

    PQ buses : |e_i - e_i^sp|            (complex P + jQ)
    PV buses : ||v_i| - |v_i^sp||  and  |P_i - P_i^sp|
    Slack    : excluded

    This is the same test the recursion uses to stop, so the solver's
    `converged` flag and `residual` can no longer disagree.

    Returns
    -------
    float
        Maximum mismatch (p.u.).
    """

    voltage = np.asarray(
        voltage,
        dtype=complex,
    )

    n_bus = problem.ybus.shape[0]

    if voltage.shape != (n_bus,):
        raise ValueError(
            "Voltage vector has an incompatible shape."
        )

    return eq14_mismatch(problem, voltage)


def solve_helm(
    problem: HelmProblem,
    config: HelmSolverConfig | None = None,
) -> HelmSolution:
    """
    Solve the HELM embedded power-flow problem.

    The coefficient recursion is delegated to
    recursion.solve_helm_series().
    """

    if config is None:
        config = HelmSolverConfig()

    if config.max_order < 1:
        raise ValueError(
            "max_order must be at least 1."
        )

    if config.min_order < 1:
        raise ValueError(
            "min_order must be at least 1."
        )

    if config.min_order > config.max_order:
        raise ValueError(
            "min_order cannot exceed max_order."
        )

    # ---------------------------------------------------------
    # Run complete HELM coefficient recursion.
    # ---------------------------------------------------------

    result = solve_helm_series(
        problem=problem,
        max_order=config.max_order,
        tol=config.tolerance,
        min_order=config.min_order,
    )

    voltage = result["voltage"]

    # ---------------------------------------------------------
    # Independently evaluate specified-power residual.
    # ---------------------------------------------------------

    residual = power_flow_residual(
        problem,
        voltage,
    )

    converged = (
        result["converged"]
        and residual <= config.tolerance
    )

    # ---------------------------------------------------------
    # Package result.
    # ---------------------------------------------------------

    return HelmSolution(
        coefficients=result[
            "voltage_coefficients"
        ],
        inverse_coefficients=result[
            "inverse_voltage_coefficients"
        ],
        reactive_power_coefficients=result[
            "reactive_power_coefficients"
        ],

        voltage=result["voltage"],
        voltage_magnitude=result[
            "voltage_magnitude"
        ],
        voltage_angle=result[
            "voltage_angle"
        ],

        active_power=result[
            "active_power"
        ],
        reactive_power=result[
            "reactive_power"
        ],

        converged=converged,
        order=result["orders_used"],
        residual=residual,

        convergence_history=result[
            "convergence_history"
        ],
    )