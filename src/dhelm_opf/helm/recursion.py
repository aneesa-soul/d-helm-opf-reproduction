"""
Holomorphic power-series recursion for AC power flow.

This module implements the coefficient recursion used by a
holomorphic embedding formulation of the AC power-flow equations.

It is intentionally independent of the RL policy and CPO algorithm.
"""

from __future__ import annotations

import numpy as np

from dhelm_opf.helm.embedding import HelmProblem


def initialize_voltage_series(
    problem: HelmProblem,
    order: int,
) -> np.ndarray:
    """
    Initialize the voltage power-series coefficients.

    V(s) = V[0] + V[1]s + V[2]s^2 + ...

    The zeroth-order voltage is initialized from the slack voltage.

    Parameters
    ----------
    problem:
        HELM problem definition.

    order:
        Highest coefficient to allocate.

    Returns
    -------
    numpy.ndarray
        Complex voltage coefficient matrix with shape:

            (order + 1, n_bus)
    """

    if order < 0:
        raise ValueError("order must be non-negative.")

    n_bus = problem.ybus.shape[0]

    coefficients = np.zeros(
        (order + 1, n_bus),
        dtype=np.complex128,
    )

    coefficients[0, :] = problem.v_slack

    return coefficients


def initialize_conjugate_series(
    voltage_coefficients: np.ndarray,
) -> np.ndarray:
    """
    Initialize coefficients for the conjugate-voltage series.

    If

        V(s) = Σ V[k] s^k,

    then the corresponding conjugate series is initialized as

        W(s) = Σ conj(V[k]) s^k.

    """

    return np.conjugate(voltage_coefficients)


def compute_series_product(
    a: np.ndarray,
    b: np.ndarray,
    k: int,
) -> complex:
    """
    Compute the kth coefficient of a Cauchy product.

    For

        A(s) = Σ a[k] s^k
        B(s) = Σ b[k] s^k,

    the kth coefficient of A(s)B(s) is

        Σ_{m=0}^{k} a[m] b[k-m].
    """

    if k < 0:
        raise ValueError("k must be non-negative.")

    if len(a) <= k or len(b) <= k:
        raise ValueError(
            "Input series do not contain enough coefficients."
        )

    return sum(
        a[m] * b[k - m]
        for m in range(k + 1)
    )


def build_embedding_rhs(
    problem: HelmProblem,
    order: int,
) -> np.ndarray:
    """
    Construct the embedded complex-power right-hand side.

    The power-flow equation is represented through a formal
    embedding parameter s.

    At s = 0, the system corresponds to the no-load/no-injection
    reference state.

    At s = 1, the full specified operating condition is recovered.

    Returns
    -------
    numpy.ndarray
        Complex RHS coefficients with shape:

            (order + 1, n_bus)
    """

    n_bus = problem.ybus.shape[0]

    rhs = np.zeros(
        (order + 1, n_bus),
        dtype=np.complex128,
    )

    if order >= 1:
        rhs[1, :] = problem.s_spec

    return rhs


def solve_linear_system(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """
    Solve a complex linear system.

    A dedicated helper keeps the numerical operation isolated from
    the coefficient bookkeeping.
    """

    matrix = np.asarray(matrix, dtype=np.complex128)
    rhs = np.asarray(rhs, dtype=np.complex128)

    return np.linalg.solve(matrix, rhs)


def enforce_slack_coefficient(
    coefficients: np.ndarray,
    problem: HelmProblem,
) -> None:
    """
    Enforce the fixed slack-bus voltage for every series coefficient.

    For a fixed reference voltage:

        V_slack(s) = V_slack

    therefore:

        V_slack[0] = V_slack
        V_slack[k] = 0,  k > 0
    """

    slack = problem.slack_bus

    coefficients[0, slack] = problem.v_slack

    if coefficients.shape[0] > 1:
        coefficients[1:, slack] = 0.0


def validate_series(
    coefficients: np.ndarray,
    problem: HelmProblem,
) -> None:
    """
    Validate basic properties of a voltage coefficient series.
    """

    if coefficients.ndim != 2:
        raise ValueError(
            "Voltage coefficients must have shape "
            "(order + 1, n_bus)."
        )

    n_bus = problem.ybus.shape[0]

    if coefficients.shape[1] != n_bus:
        raise ValueError(
            f"Expected {n_bus} buses, "
            f"got {coefficients.shape[1]}."
        )

    slack = problem.slack_bus

    if not np.isclose(
        coefficients[0, slack],
        problem.v_slack,
    ):
        raise ValueError(
            "Zeroth-order slack voltage does not match "
            "the specified reference voltage."
        )
