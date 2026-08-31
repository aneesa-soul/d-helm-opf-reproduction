import numpy as np
import pandapower.networks as pn
import pandapower as pp

from dhelm_opf.helm.problem import (
    build_helm_problem_from_network,
)
from dhelm_opf.helm.embedding import (
    HelmProblem,
)
from dhelm_opf.helm.recursion import (
    update_inverse_voltage_series,
    build_coefficient_matrix,
    build_RU_vectors,
    solve_helm_coefficient,
)


print("=" * 70)
print("D-HELM IEEE 5-BUS ORDER-1 TEST")
print("=" * 70)


# ============================================================
# 1. Load IEEE 5-bus network
# ============================================================

net = pn.case5()

print("\nRunning reference pandapower power flow...")
pp.runpp(net)

print("Reference power flow converged:", net.converged)


# ============================================================
# 2. Construct HELM problem
# ============================================================

problem = build_helm_problem_from_network(net)

print("\n" + "=" * 70)
print("HELM PROBLEM")
print("=" * 70)

print("Ybus shape:", problem.ybus.shape)
print("Slack bus :", problem.slack_bus)
print("Slack V   :", problem.v_slack)
print("PV buses  :", problem.pv_buses)
print("PQ buses  :", problem.pq_buses)

print("\nSpecified Sbus (per-unit):")

for bus, value in enumerate(problem.s_spec):
    print(
        f"bus {bus}: "
        f"P={value.real:.8f} pu, "
        f"Q={value.imag:.8f} pu"
    )


# ============================================================
# 3. Initial voltage series
# ============================================================

order = 1
n_bus = problem.ybus.shape[0]

V = np.zeros(
    (order + 1, n_bus),
    dtype=np.complex128,
)

V[0, :] = 1.0 + 0.0j

print("\nInitial V_0:")
for bus, value in enumerate(V[0]):
    print(
        f"bus {bus}: "
        f"{value.real:.8f} "
        f"{value.imag:+.8f}j"
    )


# ============================================================
# 4. Eq. (11): inverse-voltage series
# ============================================================

W = update_inverse_voltage_series(
    V,
    order,
)

print("\n" + "=" * 70)
print("EQ. (11) — W COEFFICIENTS")
print("=" * 70)

print(W)


# ============================================================
# 5. Eq. (12): coefficient matrix
# ============================================================

A = build_coefficient_matrix(problem)

print("\n" + "=" * 70)
print("EQ. (12) — COEFFICIENT MATRIX")
print("=" * 70)

print("Shape :", A.shape)
print("Rank  :", np.linalg.matrix_rank(A))
print("Cond. :", np.linalg.cond(A))
print("Det   :", np.linalg.det(A))


# ============================================================
# 6. Active-power specification
#
# IMPORTANT:
# P is already in per-unit because build_sbus()
# now converts MW -> pu.
# ============================================================

P = np.real(problem.s_spec)


# ============================================================
# 7. Reactive-power coefficient storage
#
# Q[k, bus] stores HELM reactive-power coefficients.
# These are NOT copied from pandapower generator Q.
# ============================================================

Q = np.zeros(
    (order + 1, n_bus),
    dtype=float,
)


# ============================================================
# 8. Eq. (13)
# ============================================================

R_1, U_1 = build_RU_vectors(
    problem=problem,
    voltage_coefficients=V,
    inverse_voltage_coefficients=W,
    order=order,
    active_power=P,
    reactive_power_coefficients=Q,
)

print("\n" + "=" * 70)
print("EQ. (13)")
print("=" * 70)

print("\nR_1:")
for bus, value in enumerate(R_1):
    print(
        f"bus {bus}: "
        f"{value.real:.8f} "
        f"{value.imag:+.8f}j"
    )

print("\nU_1:")
for bus, value in enumerate(U_1):
    print(
        f"bus {bus}: "
        f"{value.real:.8f} "
        f"{value.imag:+.8f}j"
    )


# ============================================================
# 9. Solve HELM order 1
# ============================================================

print("\n" + "=" * 70)
print("SOLVING HELM ORDER 1")
print("=" * 70)

V_1, Q_1 = solve_helm_coefficient(
    problem=problem,
    coefficient_matrix=A,
    voltage_coefficients=V,
    inverse_voltage_coefficients=W,
    order=order,
    active_power=P,
    reactive_power_coefficients=Q,
)


# ============================================================
# 10. Results
# ============================================================

print("\n" + "=" * 70)
print("HELM ORDER-1 RESULT")
print("=" * 70)

print("\nV_1:")

for bus, value in enumerate(V_1):
    print(
        f"bus {bus}: "
        f"{value.real:.8f} "
        f"{value.imag:+.8f}j "
        f"|V|={abs(value):.8f} pu"
    )


print("\nQ_1:")

for position, bus in enumerate(problem.pv_buses):
    print(
        f"PV bus {bus}: "
        f"Q_1={Q_1[position]:.8f} pu"
    )


print("\n" + "=" * 70)
print("PASS: IEEE 5-bus order-1 HELM calculation completed.")
print("=" * 70)
