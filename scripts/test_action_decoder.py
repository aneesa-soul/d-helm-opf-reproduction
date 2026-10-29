"""
Tests for the action decoder.

1. Bound mapping:   -1 / 0 / +1  ->  min / midpoint / max.
2. Network round trip: a decoded action written into the network shows
   up as the correct V setpoints and P injections in the HelmProblem.
3. End-to-end: random actions + random uncertainty -> HELM matches
   pandapower.
4. Wind safety: a wind unit placed at bus 0 does not overwrite the
   conventional sgen0 at the same bus.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandapower as pp

from dhelm_opf.environment.case5 import build_case5
from dhelm_opf.environment.uncertainty import (
    UncertaintyConfig,
    apply_demand_uncertainty,
    apply_wind_output,
    sample_scenario,
)
from dhelm_opf.helm.problem import build_helm_problem_from_network
from dhelm_opf.helm.solver import HelmSolverConfig, solve_helm
from dhelm_opf.models.action_config import action_config_from_network
from dhelm_opf.models.action_decoder import (
    ActionDecoder,
    apply_action_to_network,
)


def test_bounds(config, decoder) -> None:

    v_min = np.array(config.v_min)
    v_max = np.array(config.v_max)
    p_min = config.P_min
    p_max = config.P_max

    lo = decoder.decode(np.full(config.action_dim, -1.0))
    hi = decoder.decode(np.full(config.action_dim, 1.0))
    mid = decoder.decode(np.zeros(config.action_dim))

    assert np.allclose(lo.v_sp, v_min)
    assert np.allclose(hi.v_sp, v_max)
    assert np.allclose(mid.v_sp, 0.5 * (v_min + v_max))

    assert np.allclose(lo.p_g, p_min)
    assert np.allclose(hi.p_g, p_max)
    assert np.allclose(mid.p_g, 0.5 * (p_min + p_max))

    assert np.allclose(mid.p_g_pu, mid.p_g / 100.0)

    for bad in (np.zeros(7), np.full(config.action_dim, 1.5)):
        try:
            decoder.decode(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid action was accepted.")

    print("  bound mapping            : OK")


def test_round_trip(config, decoder, rng) -> None:

    net = build_case5()

    action = rng.uniform(-0.5, 0.5, size=config.action_dim)
    decoded = decoder.decode(action)

    apply_action_to_network(net, config, decoded)
    pp.runpp(net)

    problem = build_helm_problem_from_network(net)

    # Voltage setpoints land on the right buses (PV and slack).
    for bus, vm in zip(config.B_vs, decoded.v_sp):
        assert np.isclose(problem.v_sp[bus], vm), bus

    assert np.isclose(abs(problem.v_s), decoded.v_sp[
        config.B_vs.index(config.G_s.bus)
    ])

    # Generator P lands in the bus injections:
    # injection = sum(unit P on bus) - load P on bus.
    load_p = np.zeros(len(net.bus))
    for _, row in net.load.iterrows():
        load_p[int(row["bus"])] += row["p_mw"]

    unit_p = np.zeros(len(net.bus))
    for unit, p in zip(config.G_ns, decoded.p_g):
        unit_p[unit.bus] += p

    for bus in config.B_vs:
        if bus == config.G_s.bus:
            continue
        expected = (unit_p[bus] - load_p[bus]) / config.base_mva
        assert np.isclose(problem.e_sp[bus].real, expected), bus

    print("  network round trip       : OK")


def test_end_to_end(config, decoder, rng, n_cases: int = 50) -> None:

    uncertainty = UncertaintyConfig()
    worst = 0.0

    for _ in range(n_cases):

        net = build_case5()

        scenario = sample_scenario(uncertainty, rng)
        apply_demand_uncertainty(net, scenario)

        with contextlib.redirect_stdout(io.StringIO()):
            apply_wind_output(
                net,
                uncertainty.B_w,
                scenario.p_w_scale,
                uncertainty.p_w_rated_mw,
            )

        # Moderate actions keep pandapower's Newton-Raphson well inside
        # its convergence region; this test is about wiring, not range.
        action = rng.uniform(-0.6, 0.6, size=config.action_dim)
        apply_action_to_network(net, config, decoder.decode(action))

        pp.runpp(net)

        reference = net.res_bus["vm_pu"].to_numpy() * np.exp(
            1j * np.deg2rad(net.res_bus["va_degree"].to_numpy())
        )

        solution = solve_helm(
            build_helm_problem_from_network(net),
            HelmSolverConfig(n_max=60, epsilon=1e-10),
        )

        worst = max(worst, np.max(np.abs(solution.v - reference)))

    assert worst < 1e-7, worst

    print(f"  HELM vs pandapower       : OK "
          f"({n_cases} random actions, max error {worst:.2e})")


def test_wind_does_not_hijack_sgen() -> None:

    net = build_case5()
    sgen0_p = float(net.sgen.at[0, "p_mw"])

    with contextlib.redirect_stdout(io.StringIO()):
        apply_wind_output(net, (0, 4), (0.5, 0.5), (100.0, 100.0))

    assert float(net.sgen.at[0, "p_mw"]) == sgen0_p
    assert len(net.sgen) == 3

    # The derived action config must still list sgen0 and no wind unit.
    labels = [u.label for u in action_config_from_network(net).G_ns]
    assert labels == [
        "gen0@bus0", "sgen0@bus0", "gen1@bus2", "gen2@bus4"
    ], labels

    print("  wind / sgen separation   : OK")


def main() -> None:

    config = action_config_from_network(build_case5())
    decoder = ActionDecoder(config)
    rng = np.random.default_rng(0)

    print("Action decoder tests")

    test_bounds(config, decoder)
    test_round_trip(config, decoder, rng)
    test_end_to_end(config, decoder, rng)
    test_wind_does_not_hijack_sgen()

    print("All action-decoder tests passed.")


if __name__ == "__main__":
    main()
