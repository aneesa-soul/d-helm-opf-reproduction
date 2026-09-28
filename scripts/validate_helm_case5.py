"""
Validate the HELM implementation against a conventional AC power flow
for the IEEE 5-bus test system.
"""

from __future__ import annotations

import numpy as np

from dhelm_opf.environment.case5 import (
    build_case5,
    validate_case5,
    run_reference_power_flow,
)
from dhelm_opf.helm.problem import build_helm_problem_from_network
from dhelm_opf.helm.solver import HelmSolverConfig, solve_helm


def main() -> None:
    # ------------------------------------------------------------
    # 1. Build and validate IEEE 5-bus network
    # ------------------------------------------------------------
    net = build_case5()
    validate_case5(net)

    print("IEEE 5-bus network:")
    print(f"  buses      : {len(net.bus)}")
    print(f"  loads      : {len(net.load)}")
    print(f"  generators : {len(net.gen)}")
    print(f"  lines      : {len(net.line)}")

    # ------------------------------------------------------------
    # 2. Conventional AC power flow reference
    # ------------------------------------------------------------
    run_reference_power_flow(net)

    v_ac = net.res_bus.vm_pu.to_numpy()
    angle_ac = net.res_bus.va_degree.to_numpy()

    v_ac_complex = v_ac * np.exp(
        1j * np.deg2rad(angle_ac)
    )

    print("\nReference AC power flow:")
    print(f"  voltage magnitudes : {v_ac}")
    print(f"  voltage angles     : {angle_ac}")

    # ------------------------------------------------------------
    # 3. Build HELM problem
    # ------------------------------------------------------------
    problem = build_helm_problem_from_network(net)

    n_bus = problem.ybus.shape[0]

    print("\nHELM problem:")
    print(f"  number of buses : {n_bus}")
    print(f"  slack bus       : {problem.slack_bus}")
    print(f"  PV buses        : {problem.pv_buses}")
    print(f"  PQ buses        : {problem.pq_buses}")
    print(f"  slack voltage   : {problem.v_slack}")

    # ------------------------------------------------------------
    # 4. Solve HELM
    # ------------------------------------------------------------
    config = HelmSolverConfig(
        max_order=30,
        tolerance=1e-8,
        min_order=3,
    )

    solution = solve_helm(
        problem,
        config,
    )

    print("\nHELM voltage:")
    for bus in range(n_bus):
        print(
            f"  Bus {bus}: "
            f"V = {solution.voltage[bus]: .10f}, "
            f"|V| = {solution.voltage_magnitude[bus]:.10f}, "
            f"angle = {solution.voltage_angle[bus]:.10f} deg"
        )

    print("\nHELM solution:")
    print(f"  converged : {solution.converged}")
    print(f"  order     : {solution.order}")
    print(f"  residual  : {solution.residual:.6e}")

    # ------------------------------------------------------------
    # 5. Compare HELM voltage with AC reference
    # ------------------------------------------------------------
    v_helm = solution.voltage
    vm_helm = solution.voltage_magnitude
    angle_helm = solution.voltage_angle

    complex_voltage_error = np.max(
        np.abs(v_helm - v_ac_complex)
    )

    magnitude_error = np.max(
        np.abs(vm_helm - v_ac)
    )

    angle_error = np.max(
        np.abs(angle_helm - angle_ac)
    )

    print("\nHELM vs AC:")
    print(
        f"  max complex voltage error : "
        f"{complex_voltage_error:.6e}"
    )
    print(
        f"  max magnitude error       : "
        f"{magnitude_error:.6e}"
    )
    print(
        f"  max angle error (degree)  : "
        f"{angle_error:.6e}"
    )

    # ------------------------------------------------------------
    # 6. HELM power injections
    # ------------------------------------------------------------
    print("\nHELM bus injections:")

    for bus in range(n_bus):
        print(
            f"  Bus {bus}: "
            f"P = {solution.active_power[bus]: .8f} pu, "
            f"Q = {solution.reactive_power[bus]: .8f} pu"
        )

    # ------------------------------------------------------------
    # 7. Convergence history
    # ------------------------------------------------------------
    print("\nHELM convergence history:")

    for order, error in enumerate(solution.convergence_history, start=1):
        print(f"  order {order:2d}: {error:.6e}")


if __name__ == "__main__":
    main()