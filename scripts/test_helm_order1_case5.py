"""
IEEE 5-bus order-1 HELM test.

Computes the first series term (n = 1) step by step with the paper's
equations and checks it:

    Eq. (11)  W_0 = 1 / V_0 = 1          (no-load germ V_0 = 1)
    Eq. (12)  A x_1 = b_1
    Eq. (13)  R_1, U_1

Checks:
    * slack:  V_{s,1} = v_s - 1
    * PV:     Re(V_{i,1}) = U_{i,1} = (|v_i^sp|^2 - 1) / 2
    * A x_1 = b_1 holds to round-off
    * the step-by-step result equals solve_x_n() (used by the solver)
"""

import numpy as np
import pandapower as pp
import pandapower.networks as pn

from dhelm_opf.helm.problem import build_helm_problem_from_network
from dhelm_opf.helm.recursion import (
    build_A,
    build_R_U,
    compute_W,
    init_V,
    solve_x_n,
)


print("=" * 70)
print("D-HELM IEEE 5-BUS ORDER-1 TEST")
print("=" * 70)

# ============================================================
# 1. Network and HELM problem
# ============================================================

net = pn.case5()
net.ext_grid["vm_pu"] = 1.02            # non-trivial slack setpoint
pp.runpp(net)

problem = build_helm_problem_from_network(net)

print(f"B_s = {problem.B_s},  B_v = {problem.B_v},  B_q = {problem.B_q}")
print("\nSpecified e_sp (p.u.):")
for i, e_sp_i in enumerate(problem.e_sp):
    print(f"  bus {i}: p = {e_sp_i.real:+.6f}, q = {e_sp_i.imag:+.6f}")

# ============================================================
# 2. Germ and Eq. (11)
# ============================================================

n = 1
n_bus = problem.Y.shape[0]

V = init_V(problem, n)                   # V[0] = 1
Q = np.zeros((n + 1, n_bus))
W = compute_W(V, n - 1)                  # W[0] = 1

assert np.allclose(V[0], 1.0) and np.allclose(W[0], 1.0)

# ============================================================
# 3. Eqs. (12)-(13)
# ============================================================

A, Y_tr, Y_sh = build_A(problem)
P = np.real(problem.e_sp)

R_1, U_1 = build_R_U(problem=problem, n=n, V=V, W=W, P=P, Q=Q, Y_sh=Y_sh)
b_1 = np.concatenate([R_1.real, R_1.imag, U_1])

x_1 = np.linalg.solve(A, b_1)
V_1 = x_1[:n_bus] + 1j * x_1[n_bus:2 * n_bus]
Q_1 = x_1[2 * n_bus:]

print(f"\nA: shape {A.shape}, rank {np.linalg.matrix_rank(A)}, "
      f"cond {np.linalg.cond(A):.3e}")

print("\nV_1:")
for i, value in enumerate(V_1):
    print(f"  bus {i}: {value.real:+.8f} {value.imag:+.8f}j")

print("\nQ_1 (PV buses):")
for k, i in enumerate(problem.B_v):
    print(f"  bus {i}: {Q_1[k]:+.8f}")

# ============================================================
# 4. Checks
# ============================================================

s = problem.B_s
assert np.isclose(V_1[s], problem.v_s - 1.0), V_1[s]

for k, i in enumerate(problem.B_v):
    assert np.isclose(V_1[i].real, U_1[k]), (i, V_1[i].real, U_1[k])
    assert np.isclose(U_1[k], (problem.v_sp[i] ** 2 - 1.0) / 2.0)

assert np.allclose(A @ x_1, b_1, atol=1e-12)

V_1_solver, Q_1_solver = solve_x_n(problem, n, V, W, Q, A, Y_sh)
assert np.allclose(V_1_solver, V_1)
assert np.allclose(Q_1_solver[list(problem.B_v)], Q_1)

print("\n" + "=" * 70)
print("PASS: IEEE 5-bus order-1 HELM calculation verified.")
print("=" * 70)
