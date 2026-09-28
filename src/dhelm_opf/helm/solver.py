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
from .recursion import solve_helm_series


@dataclass(frozen=True)
class HelmSolverConfig:
    """Numerical settings for the HELM coefficient calculation."""

    max_order: int = 20
    tolerance: float = 1e-8
    min_order: int = 3


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
    Calculate the maximum specified power mismatch.

    PQ buses:
        P and Q are checked.

    PV buses:
        P is checked.
        Q is solved by HELM and is therefore not treated
        as a specified quantity.

    Slack bus:
        Power mismatch is not included.

    Returns
    -------
    float
        Maximum absolute specified-power mismatch in p.u.
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

    # Calculate actual complex injections:
    #
    # S = V * conj(YV)
    calculated_power = (
        voltage
        * np.conjugate(problem.ybus @ voltage)
    )

    specified = problem.s_spec

    mismatches = []

    pv_buses = set(problem.pv_buses)
    slack = problem.slack_bus

    for bus in range(n_bus):

        if bus == slack:
            continue

        # -----------------------------------------------------
        # PQ bus: P and Q are specified.
        # -----------------------------------------------------
        if bus not in pv_buses:

            p_error = (
                calculated_power[bus].real
                - specified[bus].real
            )

            q_error = (
                calculated_power[bus].imag
                - specified[bus].imag
            )

            mismatches.append(abs(p_error))
            mismatches.append(abs(q_error))

        # -----------------------------------------------------
        # PV bus: only P is specified.
        # -----------------------------------------------------
        else:

            p_error = (
                calculated_power[bus].real
                - specified[bus].real
            )

            mismatches.append(abs(p_error))

    if not mismatches:
        return 0.0

    return float(np.max(mismatches))


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