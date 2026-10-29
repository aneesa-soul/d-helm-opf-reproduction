"""
Holomorphic power-series recursion for AC power flow (NumPy reference).

This module implements the coefficient recursion of the paper's
holomorphic embedding (Eqs. 9-14, Algorithm 1).  It is intentionally
independent of the RL policy and CPO algorithm, and serves as the
reference ("oracle") for the differentiable PyTorch D-HELM.

Notation (see NOTATION.md):

    alpha              embedding parameter (alpha = 0 no load, 1 full load)
    v_i(alpha) = sum_n V_{i,n} alpha^n            (Eq. 10)
    q_i(alpha) = sum_n Q_{i,n} alpha^n            (Eq. 10)
    1/v_i*(alpha*) = sum_n W*_{i,n} (alpha*)^n    (Eq. 11)
    A x_n = b_n,  x_n = [Re V_n, Im V_n, Q_n]     (Eq. 12)
    R_n, U_n                                      (Eq. 13)

Arrays are stored as V[n, i], W[n, i], Q[n, i] (order n first).
"""

from __future__ import annotations

import numpy as np

from dhelm_opf.helm.embedding import HelmProblem


def init_V(
    problem: HelmProblem,
    n_max: int,
) -> np.ndarray:
    """
    Initialize the voltage series coefficients V[n, i].

    v(alpha) = V[0] + V[1] alpha + V[2] alpha^2 + ...

    The zeroth-order term is the paper's no-load germ:

        V[0, i] = 1  for every bus i   (W[0, i] = 1, Q[0, i] = 0)

    It is NOT the slack voltage.  The specified slack voltage enters
    through the slack row of R_n at n = 1 (V[1, s] = v_s - 1), so that
    v_s(alpha) = 1 + alpha (v_s - 1) and v_s(1) = v_s.  Every other part
    of the recursion (W[0] = 1, U_1 = (|v^sp|^2 - 1) / 2) also assumes
    V[0] = 1.

    Returns
    -------
    numpy.ndarray
        Complex array of shape (n_max + 1, n_bus).
    """

    if n_max < 0:
        raise ValueError("n_max must be non-negative.")

    n_bus = problem.Y.shape[0]

    V = np.zeros(
        (n_max + 1, n_bus),
        dtype=np.complex128,
    )

    V[0, :] = 1.0 + 0.0j

    return V


def initialize_conjugate_series(
    V: np.ndarray,
) -> np.ndarray:
    """
    Coefficients of the conjugate series:  if v(alpha) = sum V[n] alpha^n
    then its conjugate series has coefficients conj(V[n]).
    """

    return np.conjugate(V)


def compute_W(
    V,
    n_max,
):
    """
    Eq. (11): inverse-series coefficients W with v(alpha) W(alpha) = 1,

        W[0, i] = 1 / V[0, i]
        W[n, i] = - sum_{m=1}^{n} V[m, i] W[n-m, i] / V[0, i],  n >= 1

    Parameters
    ----------
    V : ndarray
        Shape (N+1, n_bus).

    n_max : int
        Highest order of W required.

    Returns
    -------
    W : ndarray
        Inverse voltage-series coefficients, same shape as V.
    """

    n_available = V.shape[0] - 1

    n_max = min(
        n_max,
        n_available,
    )

    n_bus = V.shape[1]

    W = np.zeros(
        (n_available + 1, n_bus),
        dtype=complex,
    )

    # Zeroth coefficient
    for i in range(n_bus):

        if abs(V[0, i]) < 1e-14:
            raise ValueError(
                f"V[0] is zero at bus {i}."
            )

        W[0, i] = (
            1.0 / V[0, i]
        )

    # Higher-order coefficients
    for n in range(1, n_max + 1):

        for i in range(n_bus):

            summation = 0.0 + 0.0j

            for m in range(1, n + 1):

                summation += (
                    V[m, i]
                    * W[n - m, i]
                )

            W[n, i] = (
                -summation
                / V[0, i]
            )

    return W


def build_A(problem):
    """
    Eq. (12): the real-valued coefficient matrix A.

    The bus admittance matrix is decomposed as

        Y = Y_tr + Y_sh

    where
        Y_tr : transmission matrix (row sums zero)
        Y_sh : diagonal shunt part

    In A the slack-bus row of Y_tr is replaced by the slack-voltage
    condition (Y_tr[s, j] = 0, Y_tr[s, s] = 1, paper below Eq. 12).

            [ Re(Y_tr)  -Im(Y_tr)  0 ]
        A = [ Im(Y_tr)   Re(Y_tr)  B ]
            [   B^T         0      0 ]

    with B[i, j] = 1 if bus i is the j-th PV bus.

    Returns
    -------
    A : ndarray
        Coefficient matrix of Eq. (12).

    Y_tr : ndarray
        Transmission matrix before the slack-row modification.

    Y_sh : ndarray
        Diagonal shunt admittance (vector).
    """

    Y = np.asarray(problem.Y, dtype=complex)

    n_bus = problem.Y.shape[0]
    n_v = len(problem.B_v)
    s = problem.B_s

    # ---------------------------------------------------------
    # 1. Transmission matrix Y_tr (row sums zero)
    # ---------------------------------------------------------
    Y_tr = np.zeros_like(Y, dtype=complex)

    for i in range(n_bus):
        for j in range(n_bus):
            if i != j:
                Y_tr[i, j] = Y[i, j]

    for i in range(n_bus):
        Y_tr[i, i] = -np.sum(Y_tr[i, :])

    # ---------------------------------------------------------
    # 2. Shunt part:  Y = Y_tr + Y_sh
    # ---------------------------------------------------------
    Y_sh_full = Y - Y_tr

    # The shunt contribution should be diagonal.
    off_diag = Y_sh_full.copy()
    np.fill_diagonal(off_diag, 0.0)

    if not np.allclose(off_diag, 0.0, atol=1e-10):
        raise ValueError(
            "Y_sh is not diagonal. "
            "Check the Y / Y_tr decomposition."
        )

    Y_sh = np.diag(Y_sh_full)

    # ---------------------------------------------------------
    # 3. Slack row of Y_tr
    # ---------------------------------------------------------
    Y_tr_s = Y_tr.copy()

    Y_tr_s[s, :] = 0.0
    Y_tr_s[s, s] = 1.0

    # ---------------------------------------------------------
    # 4. PV selection matrix B
    # ---------------------------------------------------------
    B = np.zeros((n_bus, n_v), dtype=float)

    for j, bus in enumerate(problem.B_v):
        B[bus, j] = 1.0

    # ---------------------------------------------------------
    # 5. Real and imaginary parts of Y_tr
    # ---------------------------------------------------------
    Y_tr_re = np.real(Y_tr_s)
    Y_tr_im = np.imag(Y_tr_s)

    # ---------------------------------------------------------
    # 6. A of Eq. (12)
    # ---------------------------------------------------------
    A = np.block([
        [Y_tr_re, -Y_tr_im, np.zeros((n_bus, n_v))],
        [Y_tr_im,  Y_tr_re, B],
        [B.T, np.zeros((n_v, n_bus)), np.zeros((n_v, n_v))]
    ])

    return A, Y_tr, Y_sh


def compute_series_product(
    a: np.ndarray,
    b: np.ndarray,
    k: int,
) -> complex:
    """
    k-th coefficient of a Cauchy product:  for A(alpha) = sum a[k] alpha^k
    and B(alpha) = sum b[k] alpha^k, returns sum_{m=0}^{k} a[m] b[k-m].
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
    n_max: int,
) -> np.ndarray:
    """
    Embedded complex-power right-hand side, alpha * e_sp: the
    injection enters at first order in alpha (Eq. 9).

    At alpha = 0 the system is the no-load reference state; at
    alpha = 1 the full specified operating condition is recovered.

    Returns
    -------
    numpy.ndarray
        Complex array of shape (n_max + 1, n_bus).
    """

    n_bus = problem.Y.shape[0]

    rhs = np.zeros(
        (n_max + 1, n_bus),
        dtype=np.complex128,
    )

    if n_max >= 1:
        rhs[1, :] = problem.e_sp

    return rhs


def solve_linear_system(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a complex linear system."""

    matrix = np.asarray(matrix, dtype=np.complex128)
    rhs = np.asarray(rhs, dtype=np.complex128)

    return np.linalg.solve(matrix, rhs)


def enforce_slack_coefficient(
    V: np.ndarray,
    problem: HelmProblem,
) -> None:
    """
    Enforce the slack-bus voltage series implied by the paper's
    embedding with the no-load germ:

        v_s(alpha) = 1 + alpha (v_s - 1)

    therefore:

        V[0, s] = 1
        V[1, s] = v_s - 1
        V[n, s] = 0,  n > 1

    The linear solve already produces these values through the slack
    row of R_n; this function only removes round-off.
    """

    s = problem.B_s

    V[0, s] = 1.0 + 0.0j

    if V.shape[0] > 1:
        V[1, s] = problem.v_s - 1.0

    if V.shape[0] > 2:
        V[2:, s] = 0.0


def validate_series(
    V: np.ndarray,
    problem: HelmProblem,
) -> None:
    """
    Validate basic properties of a voltage coefficient series.
    """

    if V.ndim != 2:
        raise ValueError(
            "V must have shape (n + 1, n_bus)."
        )

    n_bus = problem.Y.shape[0]

    if V.shape[1] != n_bus:
        raise ValueError(
            f"Expected {n_bus} buses, "
            f"got {V.shape[1]}."
        )

    s = problem.B_s

    if not np.allclose(V[0, :], 1.0):
        raise ValueError(
            "Zeroth-order coefficients must equal the no-load "
            "germ V[0] = 1 at every bus."
        )

    if V.shape[0] > 1 and not np.isclose(
        np.sum(V[:, s]),
        problem.v_s,
    ):
        raise ValueError(
            "Slack voltage series does not sum to the "
            "specified slack voltage."
        )


def build_R_U(
    problem,
    n,
    V,
    W,
    P,
    Q,
    Y_sh,
):
    """
    Eq. (13): build R_n (all buses) and U_n (PV buses), including the
    - Y_sh V_{n-1} term of b_n (Eq. 12).

        PQ  : R_{i,n} = conj(e_i^sp) conj(W_{i,n-1}) - Y_sh,i V_{i,n-1}
        PV  : R_{i,n} = P_i conj(W_{i,n-1})
                        - j sum_{m=1}^{n-1} Q_{i,m} conj(W_{i,n-m})
                        - Y_sh,i V_{i,n-1}
        slack: R_{s,1} = |v_s^sp| - 1,  R_{s,n} = 0 for n > 1

        U_{i,1} = (|v_i^sp|^2 - 1) / 2
        U_{i,n} = - sum_{m=1}^{n-1} Re(conj(V_{i,m}) V_{i,n-m}) / 2,  n > 1

    Parameters
    ----------
    problem : HelmProblem
    n : int            current series order
    V, W : ndarray     series coefficients up to order n-1
    P : ndarray        specified active injection Re(e^sp), p.u.
    Q : ndarray        reactive series coefficients Q[n, i]
    Y_sh : ndarray     diagonal shunt admittance, p.u.

    Returns
    -------
    R_n : ndarray      complex, shape (n_bus,)
    U_n : ndarray      real, shape (|B_v|,)
    """

    n_bus = problem.Y.shape[0]
    n_v = len(problem.B_v)
    s = problem.B_s

    R_n = np.zeros(n_bus, dtype=complex)
    U_n = np.zeros(n_v, dtype=float)

    pv_position = {
        bus: idx
        for idx, bus in enumerate(problem.B_v)
    }

    # =========================================================
    # R_n: PQ, PV and slack rows
    # =========================================================
    for i in range(n_bus):

        # -----------------------------------------------------
        # Slack bus
        #
        # Paper: |v_s^sp| - 1 at n = 1.  Using the complex value
        # is identical when the slack angle is 0 (the paper's
        # reference) and stays correct for a non-zero angle.
        # -----------------------------------------------------
        if i == s:

            if n == 1:
                R_n[i] = problem.v_s - 1.0
            else:
                R_n[i] = 0.0

            continue

        # -----------------------------------------------------
        # PQ bus
        # -----------------------------------------------------
        if i not in pv_position:

            e_sp_i = problem.e_sp[i]

            R_n[i] = (
                np.conjugate(e_sp_i)
                * np.conjugate(W[n - 1, i])
                - Y_sh[i]
                * V[n - 1, i]
            )

        # -----------------------------------------------------
        # PV bus
        # -----------------------------------------------------
        else:

            R_value = (
                P[i]
                * np.conjugate(
                    W[n - 1, i]
                )
            )

            if n > 1:

                q_sum = 0.0 + 0.0j

                for m in range(1, n):

                    q_sum += (
                        Q[m, i]
                        * np.conjugate(
                            W[n - m, i]
                        )
                    )

                R_value -= 1j * q_sum

            R_value -= (
                Y_sh[i]
                * V[n - 1, i]
            )

            R_n[i] = R_value

    # =========================================================
    # U_n: PV voltage-magnitude equations
    # =========================================================
    for i in problem.B_v:

        position = pv_position[i]

        if n == 1:

            v_sp_i = problem.v_sp[i]

            U_n[position] = (
                v_sp_i**2 - 1.0
            ) / 2.0

        else:

            v_sum = 0.0

            for m in range(1, n):

                v_sum += np.real(
                    np.conjugate(
                        V[m, i]
                    )
                    * V[
                        n - m, i
                    ]
                )

            U_n[position] = -0.5 * v_sum

    return R_n, U_n


def solve_x_n(
    problem,
    n,
    V,
    W,
    Q,
    A,
    Y_sh,
):
    """
    Solve one order of Eq. (12):  A x_n = b_n,

        b_n = [Re(R_n - Y_sh V_{n-1}), Im(R_n - Y_sh V_{n-1}), U_n]
        x_n = [Re(V_n), Im(V_n), Q_n]

    Returns
    -------
    V_n : ndarray
        Complex voltage coefficient of order n.

    Q_n : ndarray
        Reactive-power coefficient of order n, stored by bus index
        (zero outside B_v).
    """

    P = np.real(problem.e_sp)

    R_n, U_n = build_R_U(
        problem=problem,
        n=n,
        V=V,
        W=W,
        P=P,
        Q=Q,
        Y_sh=Y_sh,
    )

    b_n = np.concatenate([
        np.real(R_n),
        np.imag(R_n),
        U_n,
    ])

    x_n = np.linalg.solve(A, b_n)

    n_bus = problem.Y.shape[0]
    n_v = len(problem.B_v)

    V_n = (
        x_n[:n_bus]
        + 1j * x_n[n_bus:2 * n_bus]
    )

    Q_n = np.zeros(n_bus, dtype=float)

    Q_n[list(problem.B_v)] = (
        x_n[2 * n_bus:2 * n_bus + n_v]
    )

    return V_n, Q_n


def eq14_mismatch(
    problem: HelmProblem,
    v: np.ndarray,
) -> float:
    """
    Largest mismatch of the paper's termination conditions (14a)-(14c).

        (14a)  | |v_i| - |v_i^sp| |        i in B_v   (tolerance epsilon)
        (14b)  | e_i - e_i^sp |            i in B_q   (complex p + jq)
        (14c)  | Re(e_i) - Re(e_i^sp) |    i in B_v

    where e_i = v_i * sum_j conj(Y_ij v_j) is evaluated with the full Y
    at the partial-sum voltage v = sum_n V_n (alpha = 1).

    The slack bus is excluded: its voltage is fixed by construction
    and its P/Q are outputs of the critic, not specifications.
    """

    v = np.asarray(v, dtype=np.complex128)

    e = v * np.conjugate(problem.Y @ v)

    mismatches = [0.0]

    for i in problem.B_q:
        mismatches.append(
            abs(e[i] - problem.e_sp[i])
        )

    for i in problem.B_v:
        mismatches.append(
            abs(
                abs(v[i])
                - problem.v_sp[i]
            )
        )
        mismatches.append(
            abs(
                e[i].real
                - problem.e_sp[i].real
            )
        )

    return float(max(mismatches))


def solve_helm_series(
    problem,
    n_max=30,
    epsilon=1e-5,
    n_min=1,
):
    """
    Algorithm 1: compute series terms until the Eq. (14) mismatches are
    within epsilon (or n_max is reached).

    Parameters
    ----------
    problem : HelmProblem

    n_max : int
        Maximum number of series terms (Algorithm 1 / Table 1: 30).

    epsilon : float
        Tolerance on the Eq. (14a)-(14c) mismatches at alpha = 1
        (Table 1: epsilon = varepsilon = 1e-5).

    n_min : int
        Minimum number of terms before convergence can be declared.

    Returns
    -------
    result : dict
        V, W, Q        series coefficients
        v              complex voltage at alpha = 1
        v_abs, theta   magnitude (p.u.) and angle (degrees)
        e              complex injection e_i = v_i sum_j conj(Y_ij v_j)
        A, Y_tr, Y_sh  matrices of Eq. (12)
        converged, N   convergence flag and number of terms used
        increment_history, mismatch_history
    """

    n_bus = problem.Y.shape[0]

    # =========================================================
    # 1. Allocate coefficient arrays
    # =========================================================

    V = init_V(
        problem,
        n_max,
    )

    W = np.zeros(
        (n_max + 1, n_bus),
        dtype=complex,
    )

    Q = np.zeros(
        (n_max + 1, n_bus),
        dtype=float,
    )

    # =========================================================
    # 2. Coefficient matrix A and shunt vector Y_sh (Eq. 12)
    # =========================================================

    A, Y_tr, Y_sh = build_A(
        problem
    )

    # =========================================================
    # 3. Previous partial sum, for the increment diagnostic
    # =========================================================

    v_previous = np.sum(
        V[:1, :],
        axis=0,
    )

    increment_history = []
    mismatch_history = []

    converged = False
    N = 0

    # =========================================================
    # 4. Recursion (Algorithm 1, steps 2-4)
    # =========================================================

    for n in range(1, n_max + 1):

        # Eq. (11): W up to order n - 1.
        W = compute_W(
            V,
            n - 1,
        )

        # Eq. (12)-(13): solve order n.
        V_n, Q_n = solve_x_n(
            problem=problem,
            n=n,
            V=V,
            W=W,
            Q=Q,
            A=A,
            Y_sh=Y_sh,
        )

        V[n, :] = V_n
        Q[n, :] = Q_n

        enforce_slack_coefficient(
            V,
            problem,
        )

        # Partial sum at alpha = 1:  v = sum_n V_n
        v_current = np.sum(
            V[:n + 1, :],
            axis=0,
        )

        # increment : size of the newest term (diagnostic only)
        # mismatch  : paper's termination test, Eq. (14a)-(14c)
        increment = np.max(
            np.abs(v_current - v_previous)
        )

        mismatch = eq14_mismatch(problem, v_current)

        increment_history.append(increment)
        mismatch_history.append(mismatch)

        N = n

        # Algorithm 1, step 4: terminate when (14) holds.
        if n >= n_min and mismatch <= epsilon:
            converged = True
            break

        v_previous = v_current

    # =========================================================
    # 5. Final W, v, |v|, theta and e
    # =========================================================

    W = compute_W(
        V,
        N,
    )

    v = np.sum(
        V[:N + 1, :],
        axis=0,
    )

    v_abs = np.abs(v)

    theta = np.angle(
        v,
        deg=True,
    )

    e = compute_e(
        problem,
        v,
    )

    return {
        "V": V,
        "W": W,
        "Q": Q,

        "v": v,
        "v_abs": v_abs,
        "theta": theta,

        "e": e,

        "A": A,
        "Y_tr": Y_tr,
        "Y_sh": Y_sh,

        "converged": converged,
        "N": N,
        "increment_history": np.asarray(
            increment_history
        ),
        "mismatch_history": np.asarray(
            mismatch_history
        ),
    }


def compute_e(
    problem: HelmProblem,
    v: np.ndarray,
) -> np.ndarray:
    """
    Complex bus injections from the solved voltage (Algorithm 1, step 3):

        e_i = v_i * sum_j conj(Y_ij) conj(v_j)

    Real part = active injection p_i, imaginary part = reactive q_i.
    """

    Y = np.asarray(
        problem.Y,
        dtype=np.complex128,
    )

    v = np.asarray(
        v,
        dtype=np.complex128,
    )

    n_bus = Y.shape[0]

    if v.shape != (n_bus,):
        raise ValueError(
            f"v must have shape ({n_bus},), "
            f"got {v.shape}."
        )

    e = np.zeros(
        n_bus,
        dtype=np.complex128,
    )

    for i in range(n_bus):
        e[i] = (
            v[i]
            * np.sum(
                np.conjugate(Y[i, :])
                * np.conjugate(v)
            )
        )

    return e
