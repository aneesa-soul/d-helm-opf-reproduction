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

def update_inverse_voltage_series(
    voltage_coefficients: np.ndarray,
    order: int,
) -> np.ndarray:
    """
    Compute the W coefficients using the recursive relation
    from Eq. (11) of the D-HELM formulation.

    The series satisfy

        V_i(s) W_i(s) = 1

    with

        W_i,0 = 1 / V_i,0

    and for n >= 1,

        W_i,n =
            -(1 / V_i,0)
             * sum_{m=1}^{n}
               V_i,m W_i,n-m

    Parameters
    ----------
    voltage_coefficients:
        Voltage coefficients V[0], ..., V[order].

        Shape:
            (order + 1, n_bus)

    order:
        Highest coefficient to calculate.

    Returns
    -------
    numpy.ndarray
        W coefficients with shape:

            (order + 1, n_bus)
    """

    if order < 0:
        raise ValueError("order must be non-negative.")

    if voltage_coefficients.ndim != 2:
        raise ValueError(
            "Voltage coefficients must have shape "
            "(order + 1, n_bus)."
        )

    if voltage_coefficients.shape[0] <= order:
        raise ValueError(
            "Voltage coefficient array does not contain "
            f"coefficient {order}."
        )

    n_bus = voltage_coefficients.shape[1]

    W = np.zeros(
        (order + 1, n_bus),
        dtype=np.complex128,
    )

    # n = 0
    W[0, :] = 1.0 / voltage_coefficients[0, :]

    # n >= 1
    for n in range(1, order + 1):
        for i in range(n_bus):

            convolution = 0.0 + 0.0j

            for m in range(1, n + 1):
                convolution += (
                    voltage_coefficients[m, i]
                    * W[n - m, i]
                )

            W[n, i] = (
                -convolution
                / voltage_coefficients[0, i]
            )

    return W

def build_coefficient_matrix(
    problem: HelmProblem,
) -> np.ndarray:
    """
    Build the constant coefficient matrix A from Eq. (12)
    of the D-HELM paper.

    The paper defines

        A =
        [ Re(Y)   -Im(Y)    0 ]
        [ Im(Y)    Re(Y)    B ]
        [  B^T       0      0 ]

    where Y is the transmission matrix Y^tr.

    The unknown vector is

        x_n =
        [ Re(V_n) ]
        [ Im(V_n) ]
        [    Q_n  ]

    Parameters
    ----------
    problem:
        HELM problem containing:
            - Ybus
            - slack bus
            - PV bus indices

    Returns
    -------
    numpy.ndarray
        Real coefficient matrix A with shape

            (2 * n_bus + n_pv,
             2 * n_bus + n_pv)
    """

    ybus = np.asarray(
        problem.ybus,
        dtype=np.complex128,
    )

    n_bus = ybus.shape[0]

    # ------------------------------------------------------------
    # 1. Construct Y^tr
    #
    # The paper defines Y^tr by separating the shunt elements
    # from the network admittance matrix.
    #
    # For the present IEEE 5-bus line-only network, the series
    # transmission matrix can be reconstructed from the
    # off-diagonal admittances:
    #
    #     Ytr_ij = Ybus_ij, i != j
    #
    # and
    #
    #     Ytr_ii = -sum(Ytr_ij), j != i
    #
    # The remaining diagonal component represents shunt
    # admittance.
    # ------------------------------------------------------------

    y_tr = np.zeros_like(ybus)

    for i in range(n_bus):
        for j in range(n_bus):
            if i != j:
                y_tr[i, j] = ybus[i, j]

        y_tr[i, i] = -np.sum(
            y_tr[i, :]
        )

    # ------------------------------------------------------------
    # 2. Apply the slack-bus convention from Eq. (12)
    #
    # Paper:
    #
    #     Ytr_ij = 0,  i in B_s
    #     Ytr_ii = 1,  i in B_s
    #
    # ------------------------------------------------------------

    slack = problem.slack_bus

    y_tr[slack, :] = 0.0
    y_tr[slack, slack] = 1.0

    # ------------------------------------------------------------
    # 3. Build matrix B
    #
    # B is |B| x |B_v|
    #
    # B[i,j] = 1 if bus i is a PV bus corresponding to
    # PV-bus column j.
    #
    # Otherwise B[i,j] = 0.
    # ------------------------------------------------------------

    pv_buses = tuple(problem.pv_buses)

    n_pv = len(pv_buses)

    B = np.zeros(
        (n_bus, n_pv),
        dtype=float,
    )

    for j, bus in enumerate(pv_buses):
        B[bus, j] = 1.0

    # ------------------------------------------------------------
    # 4. Split Ytr into real and imaginary parts
    # ------------------------------------------------------------

    G = np.real(y_tr)
    H = np.imag(y_tr)

    # ------------------------------------------------------------
    # 5. Construct Eq. (12)
    #
    #        [ G  -H   0 ]
    #    A = [ H   G   B ]
    #        [ B^T 0   0 ]
    #
    # ------------------------------------------------------------

    zero_vq = np.zeros(
        (n_bus, n_pv),
        dtype=float,
    )

    zero_qv = np.zeros(
        (n_pv, n_bus),
        dtype=float,
    )

    zero_qq = np.zeros(
        (n_pv, n_pv),
        dtype=float,
    )

    top = np.hstack(
        (
            G,
            -H,
            zero_vq,
        )
    )

    middle = np.hstack(
        (
            H,
            G,
            B,
        )
    )

    bottom = np.hstack(
        (
            B.T,
            zero_qv,
            zero_qq,
        )
    )

    A = np.vstack(
        (
            top,
            middle,
            bottom,
        )
    )

    return A

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
