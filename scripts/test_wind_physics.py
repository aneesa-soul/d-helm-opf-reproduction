import numpy as np
import pandapower as pp

from dhelm_opf.environment.case5 import build_case5
from dhelm_opf.environment.uncertainty import (
    UncertaintyConfig,
    OperatingScenario,
    apply_wind_output,
)

from dhelm_opf.helm.problem import (
    build_sbus,
    build_helm_problem_from_network,
)
from dhelm_opf.helm.problem import build_sbus
from dhelm_opf.helm.solver import solve_helm


CONFIG = UncertaintyConfig(
    wind_buses=(1, 4),
    wind_rated_mw=(100.0, 100.0),
)


def run_case(wind_level: float):
    net = build_case5()

    scenario = OperatingScenario(
        demand_p_scale=np.ones(3),
        demand_q_scale=np.ones(3),
        wind_scale=np.array([wind_level, wind_level]),
    )

    apply_wind_output(
        net=net,
        wind_buses=CONFIG.wind_buses,
        wind_scale=scenario.wind_scale,
        wind_rated_mw=CONFIG.wind_rated_mw,
    )

    sbus = build_sbus(net)

    # Rebuild pandapower's internal power-flow/Ybus representation
    # after changing the network generation.
    pp.runpp(net, calculate_voltage_angles=True)

    problem = build_helm_problem_from_network(net)
    solution = solve_helm(problem)

    return net, sbus, solution


if __name__ == "__main__":

    for wind_level in (0.40, 0.70, 1.00):

        print("\n" + "=" * 60)
        print(f"WIND LEVEL: {wind_level:.2f}")
        print("=" * 60)

        net, sbus, solution = run_case(wind_level)

        print("\nWind generation:")
        print(net.sgen[["bus", "p_mw"]])

        print("\nSbus:")
        for i, value in enumerate(sbus):
            print(
                f"Bus {i}: "
                f"P={value.real:+.6f} p.u. "
                f"Q={value.imag:+.6f} p.u."
            )

        print("\nHELM voltage magnitudes:")
        print(solution.voltage_magnitude)

        print("\nHELM voltage angles:")
        print(solution.voltage_angle)
