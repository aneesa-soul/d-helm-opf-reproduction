"""
D-HELM solver for the IEEE 5-bus OPF reproduction.

This module combines the HELM problem representation and coefficient
recursion into a numerical solver.

Important:
    This is the first physics-solver implementation. It is intended
    to validate the HELM formulation against a conventional AC
    power-flow solution before integrating the solver with DRL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problem import HelmProblem
from .recursion import (
    initialize_voltage_series,
    enforce_slack_coefficient,
    validate_series,
)


@dataclass(frozen=True)
class HelmSolverConfig:
    """Numerical settings for the HELM coefficient calculation."""

    max_order: int = 20
    tolerance: float = 1e-10


@dataclass
class HelmSolution:
    """Result returned by the HELM solver."""

    coefficients: np.ndarray
    voltage: np.ndarray
    converged: bool
    order: int
    residual: float


def evaluate_voltage_series(
    coefficients: np.ndarray,
    z: float = 1.0,
) -> np.ndarray:
    """
    Evaluate the voltage power series at embedding parameter z.

    If

        V(z) = V[0] + V[1]z + V[2]z^2 + ...

    then this function evaluates that series at the requested z.
    """

    coefficients = np.asarray(coefficients, dtype=complex)

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
    Calculate the maximum complex power mismatch.

    S = V * conjugate(YV)

    The residual measures the difference between the specified
    injections and the injections produced by the calculated voltage.
    """

    voltage = np.asarray(voltage, dtype=complex)

    if voltage.shape != (problem.ybus.shape[0],):
        raise ValueError(
            "Voltage vector has an incompatible shape."
        )

    calculated_power = voltage * np.conjugate(
        problem.ybus @ voltage
    )

    mismatch = calculated_power - problem.s_spec

    # The slack bus is not part of the specified PQ balance.
    mismatch = mismatch.copy()
    mismatch[problem.slack_bus] = 0.0

    return float(np.max(np.abs(mismatch)))


def solve_helm(
    problem: HelmProblem,
    config: HelmSolverConfig | None = None,
) -> HelmSolution:
    """
    Solve the HELM problem.

    The current implementation establishes the solver interface and
    performs the initial coefficient construction. The coefficient
    recursion itself is delegated to recursion.py.

    Parameters
    ----------
    problem:
        HELM problem containing Ybus, Sbus, slack voltage and slack bus.

    config:
        Numerical solver configuration.

    Returns
    -------
    HelmSolution
        Voltage-series coefficients and evaluated voltage.
    """

    if config is None:
        config = HelmSolverConfig()

    if config.max_order < 1:
        raise ValueError("max_order must be at least 1.")

    coefficients = initialize_voltage_series(
        problem,
        config.max_order,
    )

    enforce_slack_coefficient(
        coefficients,
        problem,
    )

    validate_series(
        coefficients,
        problem,
    )

    voltage = evaluate_voltage_series(
        coefficients,
        z=1.0,
    )

    residual = power_flow_residual(
        problem,
        voltage,
    )

    converged = residual <= config.tolerance

    return HelmSolution(
        coefficients=coefficients,
        voltage=voltage,
        converged=converged,
        order=config.max_order,
        residual=residual,
    )
