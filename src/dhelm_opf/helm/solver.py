"""
Top-level HELM solver (NumPy reference) for the D-HELM reproduction.

This module provides the public solver interface.  The coefficient
recursion (Eqs. 10-14, Algorithm 1) is implemented in recursion.py;
this module calls it and packages the result into a HelmSolution.

Notation follows the paper (see NOTATION.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problem import HelmProblem
from .recursion import eq14_mismatch, solve_helm_series


@dataclass(frozen=True)
class HelmSolverConfig:
    """
    Numerical settings of Algorithm 1.

    Defaults follow Table 1 of the paper:
        n_max = 30,  epsilon = varepsilon = 1e-5  (Eq. 14 mismatches)
    """

    n_max: int = 30
    epsilon: float = 1e-5
    n_min: int = 1


@dataclass
class HelmSolution:
    """Result returned by the HELM solver."""

    V: np.ndarray          # V[n, i]   voltage series coefficients
    W: np.ndarray          # W[n, i]   inverse series coefficients (Eq. 11)
    Q: np.ndarray          # Q[n, i]   reactive series coefficients

    v: np.ndarray          # complex bus voltage at alpha = 1
    v_abs: np.ndarray      # |v_i|
    theta: np.ndarray      # theta_i, degrees

    e: np.ndarray          # complex injection e_i = p_i + j q_i

    converged: bool
    N: int                 # number of series terms used
    mismatch: float        # Eq. (14) mismatch of the returned v

    increment_history: np.ndarray


def evaluate_voltage_series(
    V: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Evaluate the voltage series at embedding parameter alpha (Eq. 10):

        v(alpha) = V[0] + V[1] alpha + V[2] alpha^2 + ...
    """

    V = np.asarray(
        V,
        dtype=complex,
    )

    if V.ndim != 2:
        raise ValueError(
            "V must have shape (n + 1, n_bus)."
        )

    v = np.zeros(
        V.shape[1],
        dtype=complex,
    )

    for n in range(V.shape[0]):
        v += V[n] * alpha**n

    return v


def power_flow_mismatch(
    problem: HelmProblem,
    v: np.ndarray,
) -> float:
    """
    Maximum mismatch of the paper's termination conditions (14a)-(14c).

    PQ buses : |e_i - e_i^sp|            (complex p + jq)
    PV buses : ||v_i| - |v_i^sp||  and  |p_i - p_i^sp|
    Slack    : excluded

    This is the same test the recursion uses to stop, so the solver's
    `converged` flag and `mismatch` cannot disagree.
    """

    v = np.asarray(
        v,
        dtype=complex,
    )

    n_bus = problem.Y.shape[0]

    if v.shape != (n_bus,):
        raise ValueError(
            "Voltage vector has an incompatible shape."
        )

    return eq14_mismatch(problem, v)


def solve_helm(
    problem: HelmProblem,
    config: HelmSolverConfig | None = None,
) -> HelmSolution:
    """
    Solve the HELM embedded power-flow problem (Algorithm 1).

    The coefficient recursion is delegated to
    recursion.solve_helm_series().
    """

    if config is None:
        config = HelmSolverConfig()

    if config.n_max < 1:
        raise ValueError(
            "n_max must be at least 1."
        )

    if config.n_min < 1:
        raise ValueError(
            "n_min must be at least 1."
        )

    if config.n_min > config.n_max:
        raise ValueError(
            "n_min cannot exceed n_max."
        )

    result = solve_helm_series(
        problem=problem,
        n_max=config.n_max,
        epsilon=config.epsilon,
        n_min=config.n_min,
    )

    v = result["v"]

    # Independently re-evaluate the Eq. (14) mismatch of the result.
    mismatch = power_flow_mismatch(
        problem,
        v,
    )

    converged = (
        result["converged"]
        and mismatch <= config.epsilon
    )

    return HelmSolution(
        V=result["V"],
        W=result["W"],
        Q=result["Q"],

        v=v,
        v_abs=result["v_abs"],
        theta=result["theta"],

        e=result["e"],

        converged=converged,
        N=result["N"],
        mismatch=mismatch,

        increment_history=result["increment_history"],
    )
