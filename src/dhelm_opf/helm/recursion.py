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

def build_RU_vectors(
    problem: HelmProblem,
    voltage_coefficients: np.ndarray,
    inverse_voltage_coefficients: np.ndarray,
    order: int,
    active_power: np.ndarray,
    reactive_power_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct R_n and U_n according to Eq. (13) of the D-HELM paper.

    Parameters
    ----------
    problem:
        HELM problem containing bus classifications and voltage
        setpoints.

    voltage_coefficients:
        Voltage power-series coefficients V[n, i].

        Shape:
            (order + 1, n_bus)

    inverse_voltage_coefficients:
        Inverse/conjugate-voltage series coefficients W[n, i].

        Shape:
            (order + 1, n_bus)

    order:
        Current HELM coefficient order n.

        Must satisfy:
            order >= 1

    active_power:
        Specified active-power vector P_i.

        Shape:
            (n_bus,)

    reactive_power_coefficients:
        Reactive-power series coefficients Q[n, i].

        Shape:
            (order + 1, n_bus)

        Row 0 is unused and should normally be zero.

    Returns
    -------
    R_n:
        Complex vector of length n_bus.

    U_n:
        Real-valued vector containing one entry for every PV bus.

    Notes
    -----
    This implements Eq. (13):

        PQ:
            R_i,n = e_i* W_i,n-1*

        PV:
            R_i,n =
                P_i W_i,n-1*
                - j sum_{m=1}^{n-1}
                    Q_i,m W_i,n-m*

        Slack:
            R_i,1 = |v_i|^sp - 1
            R_i,n = 0, n > 1

        PV voltage equation:

            U_i,1 =
                (|v_i^sp|^2 - 1) / 2

            U_i,n =
                -1/2 sum_{m=1}^{n-1}
                    V_i,m* V_i,n-m
    """

    # ------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------

    if order < 1:
        raise ValueError(
            "HELM coefficient order must be >= 1."
        )

    n_bus = problem.ybus.shape[0]

    voltage_coefficients = np.asarray(
        voltage_coefficients,
        dtype=np.complex128,
    )

    inverse_voltage_coefficients = np.asarray(
        inverse_voltage_coefficients,
        dtype=np.complex128,
    )

    active_power = np.asarray(
        active_power,
        dtype=float,
    )

    reactive_power_coefficients = np.asarray(
        reactive_power_coefficients,
        dtype=float,
    )

    expected_shape = (order + 1, n_bus)

    if voltage_coefficients.shape != expected_shape:
        raise ValueError(
            "voltage_coefficients must have shape "
            f"{expected_shape}, got "
            f"{voltage_coefficients.shape}."
        )

    if inverse_voltage_coefficients.shape != expected_shape:
        raise ValueError(
            "inverse_voltage_coefficients must have shape "
            f"{expected_shape}, got "
            f"{inverse_voltage_coefficients.shape}."
        )

    if reactive_power_coefficients.shape != expected_shape:
        raise ValueError(
            "reactive_power_coefficients must have shape "
            f"{expected_shape}, got "
            f"{reactive_power_coefficients.shape}."
        )

    if active_power.shape != (n_bus,):
        raise ValueError(
            f"active_power must have shape ({n_bus},)."
        )

    # ------------------------------------------------------------
    # Allocate Eq. (13) vectors
    # ------------------------------------------------------------

    R_n = np.zeros(
        n_bus,
        dtype=np.complex128,
    )

    U_n = np.zeros(
        len(problem.pv_buses),
        dtype=float,
    )

    # ------------------------------------------------------------
    # PQ buses
    #
    # R_i,n = e_i* W_i,n-1*
    # ------------------------------------------------------------

    for bus in problem.pq_buses:

        e_spec = problem.s_spec[bus]

        R_n[bus] = (
            np.conjugate(e_spec)
            * np.conjugate(
                inverse_voltage_coefficients[order - 1, bus]
            )
        )

    # ------------------------------------------------------------
    # PV buses
    #
    # R_i,n =
    #     P_i W_i,n-1*
    #     - j sum(Q_i,m W_i,n-m*)
    # ------------------------------------------------------------

    for bus in problem.pv_buses:

        # First term:
        #
        # P_i W_i,n-1*
        #
        r_value = (
            active_power[bus]
            * np.conjugate(
                inverse_voltage_coefficients[
                    order - 1,
                    bus,
                ]
            )
        )

        # Second term:
        #
        # -j sum_{m=1}^{n-1}
        #       Q_i,m W_i,n-m*
        #
        if order > 1:

            q_sum = 0.0 + 0.0j

            for m in range(1, order):

                q_m = reactive_power_coefficients[
                    m,
                    bus,
                ]

                w_term = np.conjugate(
                    inverse_voltage_coefficients[
                        order - m,
                        bus,
                    ]
                )

                q_sum += q_m * w_term

            r_value -= 1j * q_sum

        R_n[bus] = r_value

    # ------------------------------------------------------------
    # Slack bus
    #
    # R_i,1 = |v_i^sp| - 1
    #
    # R_i,n = 0, n > 1
    # ------------------------------------------------------------

    slack = problem.slack_bus

    if order == 1:

        R_n[slack] = (
            abs(problem.v_slack) - 1.0
        )

    else:

        R_n[slack] = 0.0

    # ------------------------------------------------------------
    # U_n for PV buses
    #
    # n = 1:
    #
    # U_i,1 =
    #     (|v_i^sp|^2 - 1) / 2
    #
    # n > 1:
    #
    # U_i,n =
    #     -1/2 sum V_i,m* V_i,n-m
    # ------------------------------------------------------------

    for pv_position, bus in enumerate(problem.pv_buses):

        if order == 1:

            voltage_setpoint = (
                problem.voltage_setpoints[bus]
            )

            U_n[pv_position] = (
                voltage_setpoint**2 - 1.0
            ) / 2.0

        else:

            u_sum = 0.0 + 0.0j

            for m in range(1, order):

                v_m = np.conjugate(
                    voltage_coefficients[
                        m,
                        bus,
                    ]
                )

                v_n_minus_m = (
                    voltage_coefficients[
                        order - m,
                        bus,
                    ]
                )

                u_sum += (
                    v_m
                    * v_n_minus_m
                )

            U_n[pv_position] = (
                -0.5 * u_sum.real
            )

    return R_n, U_n

def solve_coefficient_order(
    problem: HelmProblem,
    voltage_coefficients: np.ndarray,
    inverse_voltage_coefficients: np.ndarray,
    order: int,
    active_power: np.ndarray,
    reactive_power_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the HELM coefficient system for order n.

    This implements the linear solve associated with
    Eqs. (12) and (13) of the D-HELM formulation.

    The system solved is

        A x_n = b_n

    where

        x_n =
            [ Re(V_n)
              Im(V_n)
              Q_n ]

    and

        b_n =
            [ Re(R_n)
              Im(R_n)
              U_n ].

    Returns
    -------
    voltage_n:
        Complex voltage coefficient V_n.

    reactive_n:
        Reactive-power coefficient Q_n for PV buses.
    """

    if order < 1:
        raise ValueError(
            "HELM coefficient order must be >= 1."
        )

    n_bus = problem.ybus.shape[0]
    n_pv = len(problem.pv_buses)

    # ------------------------------------------------------------
    # Eq. (12): constant coefficient matrix
    # ------------------------------------------------------------

    A = build_coefficient_matrix(problem)

    # ------------------------------------------------------------
    # Eq. (13): construct R_n and U_n
    # ------------------------------------------------------------

    R_n, U_n = build_RU_vectors(
        problem=problem,
        voltage_coefficients=voltage_coefficients,
        inverse_voltage_coefficients=inverse_voltage_coefficients,
        order=order,
        active_power=active_power,
        reactive_power_coefficients=reactive_power_coefficients,
    )

    # ------------------------------------------------------------
    # Construct the RHS of Eq. (12)
    #
    # [ Re(R_n) ]
    # [ Im(R_n) ]
    # [   U_n   ]
    # ------------------------------------------------------------

    rhs = np.concatenate(
        [
            R_n.real,
            R_n.imag,
            U_n,
        ]
    )

    expected_size = 2 * n_bus + n_pv

    if A.shape != (expected_size, expected_size):
        raise ValueError(
            "Coefficient matrix has unexpected shape: "
            f"{A.shape}, expected "
            f"({expected_size}, {expected_size})."
        )

    if rhs.shape != (expected_size,):
        raise ValueError(
            "Coefficient RHS has unexpected shape: "
            f"{rhs.shape}, expected "
            f"({expected_size},)."
        )

    # ------------------------------------------------------------
    # Solve Eq. (12)
    # ------------------------------------------------------------

    solution = np.linalg.solve(A, rhs)

    # ------------------------------------------------------------
    # Extract V_n
    #
    # First n_bus entries:
    #     Re(V_n)
    #
    # Next n_bus entries:
    #     Im(V_n)
    # ------------------------------------------------------------

    voltage_n = (
        solution[:n_bus]
        + 1j * solution[n_bus:2 * n_bus]
    )

    # ------------------------------------------------------------
    # Extract Q_n
    #
    # Last n_pv entries correspond to PV buses.
    # ------------------------------------------------------------

    reactive_n = solution[
        2 * n_bus:
    ]

    return voltage_n, reactive_n

def solve_helm_coefficient(
    problem: HelmProblem,
    coefficient_matrix: np.ndarray,
    voltage_coefficients: np.ndarray,
    inverse_voltage_coefficients: np.ndarray,
    order: int,
    active_power: np.ndarray,
    reactive_power_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the HELM linear system for one coefficient order.

    Implements the coefficient solve

        A x_n =
            [ Re(R_n) ]
            [ Im(R_n) ]
            [    U_n  ]

    where

        x_n =
            [ Re(V_n) ]
            [ Im(V_n) ]
            [    Q_n  ]

    Eq. (12) provides A and Eq. (13) provides R_n and U_n.

    Parameters
    ----------
    problem:
        HELM problem definition.

    coefficient_matrix:
        Constant coefficient matrix A from Eq. (12).

    voltage_coefficients:
        Voltage series coefficients calculated up to the
        current order.

    inverse_voltage_coefficients:
        Inverse-voltage series coefficients calculated up to
        the current order.

    order:
        Coefficient order n being solved.

    active_power:
        Specified active-power vector.

    reactive_power_coefficients:
        Reactive-power coefficient matrix.

    Returns
    -------
    voltage_n:
        Complex voltage coefficient V_n.

    reactive_power_n:
        Reactive-power coefficient Q_n for the PV buses.
    """

    if order < 1:
        raise ValueError(
            "HELM coefficient order must be >= 1."
        )

    n_bus = problem.ybus.shape[0]
    n_pv = len(problem.pv_buses)

    expected_size = 2 * n_bus + n_pv

    coefficient_matrix = np.asarray(
        coefficient_matrix,
        dtype=float,
    )

    if coefficient_matrix.shape != (
        expected_size,
        expected_size,
    ):
        raise ValueError(
            "Coefficient matrix must have shape "
            f"({expected_size}, {expected_size}), "
            f"got {coefficient_matrix.shape}."
        )

    # ------------------------------------------------------------
    # Eq. (13)
    # ------------------------------------------------------------

    R_n, U_n = build_RU_vectors(
        problem=problem,
        voltage_coefficients=voltage_coefficients,
        inverse_voltage_coefficients=inverse_voltage_coefficients,
        order=order,
        active_power=active_power,
        reactive_power_coefficients=reactive_power_coefficients,
    )

    # ------------------------------------------------------------
    # Construct RHS of Eq. (12)
    #
    #       [ Re(R_n) ]
    # b_n = [ Im(R_n) ]
    #       [   U_n   ]
    # ------------------------------------------------------------

    rhs = np.concatenate(
        [
            R_n.real,
            R_n.imag,
            U_n,
        ]
    )

    # ------------------------------------------------------------
    # Solve
    #
    # A x_n = b_n
    # ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("HELM LINEAR SYSTEM DIAGNOSTIC")
    print("=" * 60)

    print("order:", order)
    print("matrix shape:", coefficient_matrix.shape)
    print("matrix rank:", np.linalg.matrix_rank(coefficient_matrix))
    print("matrix condition number:", np.linalg.cond(coefficient_matrix))
    print("matrix determinant:", np.linalg.det(coefficient_matrix))

    print("\nRHS:")
    print(rhs)

    print("\nCoefficient matrix:")
    print(coefficient_matrix)
    
    x_n = np.linalg.solve(
        coefficient_matrix,
        rhs,
    )

    # ------------------------------------------------------------
    # Extract V_n
    #
    # x_n =
    #
    # [ Re(V_n) ]
    # [ Im(V_n) ]
    # [   Q_n   ]
    # ------------------------------------------------------------

    voltage_n = (
        x_n[:n_bus]
        + 1j * x_n[n_bus:2 * n_bus]
    )

    # ------------------------------------------------------------
    # Extract Q_n
    # ------------------------------------------------------------

    reactive_power_n = (
        x_n[2 * n_bus:]
    )

    return voltage_n, reactive_power_n