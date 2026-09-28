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
    voltage_coefficients,
    max_order,
):
    """
    Compute inverse voltage-series coefficients W such that

        V(s) W(s) = 1.

    Parameters
    ----------
    voltage_coefficients : ndarray
        Shape (N+1, n_bus).

    max_order : int
        Highest inverse-series order required.

    Returns
    -------
    W : ndarray
        Inverse voltage-series coefficients.
    """

    max_available = voltage_coefficients.shape[0] - 1

    max_order = min(
        max_order,
        max_available,
    )

    n_bus = voltage_coefficients.shape[1]

    W = np.zeros(
        (max_available + 1, n_bus),
        dtype=complex,
    )

    # Zeroth coefficient
    for bus in range(n_bus):

        if abs(voltage_coefficients[0, bus]) < 1e-14:
            raise ValueError(
                f"V(0) is zero at bus {bus}."
            )

        W[0, bus] = (
            1.0 / voltage_coefficients[0, bus]
        )

    # Higher-order coefficients
    for order in range(1, max_order + 1):

        for bus in range(n_bus):

            summation = 0.0 + 0.0j

            for m in range(1, order + 1):

                summation += (
                    voltage_coefficients[m, bus]
                    * W[order - m, bus]
                )

            W[order, bus] = (
                -summation
                / voltage_coefficients[0, bus]
            )

    return W

def build_coefficient_matrix(problem):
    """
    Build the real-valued HELM coefficient matrix A.

    The network admittance matrix is decomposed as:

        Ybus = Ytr_raw + Ysh

    where:
        Ytr_raw : transmission/network series contribution
        Ysh     : diagonal shunt contribution

    The slack-bus row of Ytr_raw is then replaced by the
    slack-voltage constraint before constructing A.

    Returns
    -------
    A : ndarray
        HELM coefficient matrix.

    y_tr_raw : ndarray
        Transmission matrix before slack-row modification.

    y_sh_diag : ndarray
        Diagonal shunt admittance vector.
    """

    ybus = np.asarray(problem.ybus, dtype=complex)

    n_bus = problem.ybus.shape[0]
    n_pv = len(problem.pv_buses)
    slack = problem.slack_bus

    # ---------------------------------------------------------
    # 1. Construct raw transmission matrix
    # ---------------------------------------------------------
    y_tr_raw = np.zeros_like(ybus, dtype=complex)

    for i in range(n_bus):
        for j in range(n_bus):
            if i != j:
                y_tr_raw[i, j] = ybus[i, j]

    for i in range(n_bus):
        y_tr_raw[i, i] = -np.sum(y_tr_raw[i, :])

    # ---------------------------------------------------------
    # 2. Extract shunt contribution
    #
    # Ybus = Ytr_raw + Ysh
    # ---------------------------------------------------------
    y_sh = ybus - y_tr_raw

    # The shunt contribution should be diagonal.
    off_diag = y_sh.copy()
    np.fill_diagonal(off_diag, 0.0)

    if not np.allclose(off_diag, 0.0, atol=1e-10):
        raise ValueError(
            "Ysh is not diagonal. "
            "Check the Ybus/transmission decomposition."
        )

    y_sh_diag = np.diag(y_sh)

    # ---------------------------------------------------------
    # 3. Apply slack-bus constraint to transmission matrix
    # ---------------------------------------------------------
    y_tr = y_tr_raw.copy()

    y_tr[slack, :] = 0.0
    y_tr[slack, slack] = 1.0

    # ---------------------------------------------------------
    # 4. PV-bus Q coefficient selector matrix B
    # ---------------------------------------------------------
    B = np.zeros((n_bus, n_pv), dtype=float)

    for j, bus in enumerate(problem.pv_buses):
        B[bus, j] = 1.0

    # ---------------------------------------------------------
    # 5. Split complex Ytr into G + jH
    # ---------------------------------------------------------
    G = np.real(y_tr)
    H = np.imag(y_tr)

    # ---------------------------------------------------------
    # 6. HELM real coefficient matrix
    #
    #       [ G  -H   0 ]
    # A =   [ H   G   B ]
    #       [ Bᵀ  0   0 ]
    # ---------------------------------------------------------
    A = np.block([
        [G,       -H,       np.zeros((n_bus, n_pv))],
        [H,        G,        B],
        [B.T, np.zeros((n_pv, n_bus)), np.zeros((n_pv, n_pv))]
    ])

    return A, y_tr_raw, y_sh_diag

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
    problem,
    order,
    voltage_coefficients,
    inverse_voltage_coefficients,
    active_power,
    reactive_power_coefficients,
    y_sh_diag,
):
    """
    Build the HELM R_n and U_n coefficient vectors.

    Parameters
    ----------
    problem : HELMProblem
        HELM problem definition.

    order : int
        Current series order n.

    voltage_coefficients : ndarray
        Shape (order+1, n_bus).

    inverse_voltage_coefficients : ndarray
        Voltage inverse-series coefficients W.

    active_power : ndarray
        Specified active-power injections in p.u.

    reactive_power_coefficients : ndarray
        Shape (order+1, n_bus).

    y_sh_diag : ndarray
        Diagonal shunt admittance in p.u.

    Returns
    -------
    R_n : ndarray
        Complex RHS vector for the voltage equations.

    U_n : ndarray
        Real RHS vector for the PV voltage-magnitude equations.
    """

    n_bus = problem.ybus.shape[0]
    n_pv = len(problem.pv_buses)
    slack = problem.slack_bus

    R_n = np.zeros(n_bus, dtype=complex)
    U_n = np.zeros(n_pv, dtype=float)

    pv_position = {
        bus: idx
        for idx, bus in enumerate(problem.pv_buses)
    }

    # =========================================================
    # PQ and PV bus equations
    # =========================================================
    for bus in range(n_bus):

        # -----------------------------------------------------
        # Slack bus
        # -----------------------------------------------------
        if bus == slack:

            if order == 1:
                R_n[bus] = abs(problem.v_slack) - 1.0
            else:
                R_n[bus] = 0.0

            continue

        # -----------------------------------------------------
        # PQ bus
        #
        # R(i,n) =
        #   conj(S_spec(i)) conj(W(i,n-1))
        #   - Ysh(i) V(i,n-1)
        # -----------------------------------------------------
        if bus not in pv_position:

            s_spec = problem.s_spec[bus]

            R_n[bus] = (
                np.conjugate(s_spec)
                * np.conjugate(inverse_voltage_coefficients[order - 1, bus])
                - y_sh_diag[bus]
                * voltage_coefficients[order - 1, bus]
            )

        # -----------------------------------------------------
        # PV bus
        #
        # R(i,n) =
        #   P(i) conj(W(i,n-1))
        #   - j Σ Q(i,m) conj(W(i,n-m))
        #   - Ysh(i) V(i,n-1)
        # -----------------------------------------------------
        else:

            r_value = (
                active_power[bus]
                * np.conjugate(
                    inverse_voltage_coefficients[order - 1, bus]
                )
            )

            if order > 1:

                q_sum = 0.0 + 0.0j

                for m in range(1, order):

                    q_sum += (
                        reactive_power_coefficients[m, bus]
                        * np.conjugate(
                            inverse_voltage_coefficients[
                                order - m, bus
                            ]
                        )
                    )

                r_value -= 1j * q_sum

            r_value -= (
                y_sh_diag[bus]
                * voltage_coefficients[order - 1, bus]
            )

            R_n[bus] = r_value

    # =========================================================
    # PV voltage-magnitude equations
    # =========================================================
    for bus in problem.pv_buses:

        position = pv_position[bus]

        if order == 1:

            voltage_setpoint = problem.voltage_setpoints[bus]

            U_n[position] = (
                voltage_setpoint**2 - 1.0
            ) / 2.0

        else:

            voltage_sum = 0.0

            for m in range(1, order):

                voltage_sum += np.real(
                    np.conjugate(
                        voltage_coefficients[m, bus]
                    )
                    * voltage_coefficients[
                        order - m, bus
                    ]
                )

            U_n[position] = -0.5 * voltage_sum

    return R_n, U_n

def solve_coefficient_order(
    problem,
    order,
    voltage_coefficients,
    inverse_voltage_coefficients,
    reactive_power_coefficients,
    A,
    y_sh_diag,
):
    """
    Solve one HELM series coefficient order.

    Returns
    -------
    voltage_n : ndarray
        Complex voltage coefficient V(n).

    q_n : ndarray
        Reactive-power coefficient Q(n), stored by bus index.
    """

    active_power = np.real(problem.s_spec)

    R_n, U_n = build_RU_vectors(
        problem=problem,
        order=order,
        voltage_coefficients=voltage_coefficients,
        inverse_voltage_coefficients=inverse_voltage_coefficients,
        active_power=active_power,
        reactive_power_coefficients=reactive_power_coefficients,
        y_sh_diag=y_sh_diag,
    )

    rhs = np.concatenate([
        np.real(R_n),
        np.imag(R_n),
        U_n,
    ])

    solution = np.linalg.solve(A, rhs)

    n_bus = problem.ybus.shape[0]
    n_pv = len(problem.pv_buses)

    voltage_n = (
        solution[:n_bus]
        + 1j * solution[n_bus:2 * n_bus]
    )

    q_n = np.zeros(n_bus, dtype=float)

    q_n[list(problem.pv_buses)] = (
        solution[2 * n_bus:2 * n_bus + n_pv]
    )

    return voltage_n, q_n

def solve_helm_series(
    problem,
    max_order=20,
    tol=1e-8,
    min_order=3,
):
    """
    Solve the HELM embedded power-flow problem
    by recursively computing voltage-series coefficients.

    Parameters
    ----------
    problem : HELMProblem
        HELM problem definition.

    max_order : int
        Maximum number of series coefficients.

    tol : float
        Convergence tolerance based on successive
        partial sums evaluated at s = 1.

    min_order : int
        Minimum number of orders before convergence
        can be declared.

    Returns
    -------
    result : dict
        Dictionary containing the HELM solution and
        convergence information.
    """

    n_bus = problem.ybus.shape[0]

    # =========================================================
    # 1. Allocate coefficient arrays
    # =========================================================

    V_coeff = initialize_voltage_series(
        problem,
        max_order,
    )

    W_coeff = np.zeros(
        (max_order + 1, n_bus),
        dtype=complex,
    )

    Q_coeff = np.zeros(
        (max_order + 1, n_bus),
        dtype=float,
    )

    # =========================================================
    # 2. Build HELM coefficient matrix and shunt vector
    # =========================================================

    A, y_tr_raw, y_sh_diag = build_coefficient_matrix(
        problem
    )

    # =========================================================
    # 3. Previous partial voltage for convergence test
    # =========================================================

    V_previous = np.sum(
        V_coeff[:1, :],
        axis=0,
    )

    convergence_history = []

    converged = False
    orders_used = 0

    # =========================================================
    # 4. Recursive coefficient calculation
    # =========================================================

    for order in range(1, max_order + 1):

        # -----------------------------------------------------
        # Compute inverse voltage series coefficients up to
        # order - 1.
        # -----------------------------------------------------
        W_coeff = update_inverse_voltage_series(
            V_coeff,
            order - 1,
        )

        # -----------------------------------------------------
        # Solve current coefficient order.
        # -----------------------------------------------------
        V_n, Q_n = solve_coefficient_order(
            problem=problem,
            order=order,
            voltage_coefficients=V_coeff,
            inverse_voltage_coefficients=W_coeff,
            reactive_power_coefficients=Q_coeff,
            A=A,
            y_sh_diag=y_sh_diag,
        )

        # -----------------------------------------------------
        # Store coefficients.
        # -----------------------------------------------------
        V_coeff[order, :] = V_n
        Q_coeff[order, :] = Q_n

        # -----------------------------------------------------
        # Enforce slack coefficient.
        # -----------------------------------------------------
        enforce_slack_coefficient(
            V_coeff,
            problem,
        )

        # -----------------------------------------------------
        # Partial series evaluated at s = 1.
        #
        # V(1) ≈ Σ V(n)
        # -----------------------------------------------------
        V_current = np.sum(
            V_coeff[:order + 1, :],
            axis=0,
        )

        # -----------------------------------------------------
        # Convergence measure.
        # -----------------------------------------------------
        error = np.max(
            np.abs(V_current - V_previous)
        )

        convergence_history.append(error)

        orders_used = order

        # -----------------------------------------------------
        # Check convergence.
        # -----------------------------------------------------
        if order >= min_order and error < tol:
            converged = True
            break

        V_previous = V_current

    # =========================================================
    # 5. Final inverse series
    # =========================================================

    W_coeff = update_inverse_voltage_series(
        V_coeff,
        orders_used,
    )

    # =========================================================
    # 6. Final voltage estimate
    # =========================================================

    voltage = np.sum(
        V_coeff[:orders_used + 1, :],
        axis=0,
    )

    # =========================================================
    # 7. Voltage magnitude and angle
    # =========================================================

    voltage_magnitude = np.abs(voltage)

    voltage_angle = np.angle(
        voltage,
        deg=True,
    )

    # =========================================================
    # 8. Calculate resulting complex power injections
    # =========================================================

    injections = compute_bus_injections(
        problem,
        voltage,
    )

    active_power = np.real(injections)
    reactive_power = np.imag(injections)

    # =========================================================
    # 9. Return complete result
    # =========================================================

    return {
        "voltage_coefficients": V_coeff,
        "inverse_voltage_coefficients": W_coeff,
        "reactive_power_coefficients": Q_coeff,

        "voltage": voltage,
        "voltage_magnitude": voltage_magnitude,
        "voltage_angle": voltage_angle,

        "active_power": active_power,
        "reactive_power": reactive_power,

        "A": A,
        "y_tr_raw": y_tr_raw,
        "y_sh_diag": y_sh_diag,

        "converged": converged,
        "orders_used": orders_used,
        "convergence_history": np.asarray(
            convergence_history
        ),
    }

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

def compute_bus_injections(
    problem: HelmProblem,
    voltage: np.ndarray,
) -> np.ndarray:
    """
    Compute complex bus power injections from the solved
    HELM voltage vector.

    Implements the physical power-injection calculation used
    in Algorithm 1 of the D-HELM paper:

        e_i = V_i * sum_j conj(Y_ij) * conj(V_j)

    Parameters
    ----------
    problem:
        HELM problem containing the full network admittance matrix.

    voltage:
        Complex bus-voltage vector evaluated at s = 1.

    Returns
    -------
    np.ndarray
        Complex bus-injection vector e.

        Real part  = active-power injection P
        Imaginary part = reactive-power injection Q
    """

    ybus = np.asarray(
        problem.ybus,
        dtype=np.complex128,
    )

    voltage = np.asarray(
        voltage,
        dtype=np.complex128,
    )

    n_bus = ybus.shape[0]

    if voltage.shape != (n_bus,):
        raise ValueError(
            f"voltage must have shape ({n_bus},), "
            f"got {voltage.shape}."
        )

    injections = np.zeros(
        n_bus,
        dtype=np.complex128,
    )

    for i in range(n_bus):
        injections[i] = (
            voltage[i]
            * np.sum(
                np.conjugate(ybus[i, :])
                * np.conjugate(voltage)
            )
        )

    return injections