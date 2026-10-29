"""
Validate the NumPy HELM solver against pandapower over random scenarios.

This is the reference ("oracle") test for the HELM forward pass.  The
later differentiable PyTorch D-HELM will be checked against this solver,
so this solver must first match a conventional AC power flow.

Each scenario randomizes:
    - slack voltage magnitude         U(0.95, 1.05)  p.u.
    - PV-bus voltage setpoints        U(0.95, 1.05)  p.u. (each bus)
    - demand P and Q scaling          paper range [0.60, 1.10]
    - wind scaling                    paper range [0.40, 1.00]

Two checks are made:
    1. Tight tolerance (1e-10): HELM must match pandapower to ~1e-8,
       which shows the embedding itself is correct.
    2. Paper tolerance (Algorithm 1: 1e-5, n_max = 30): every scenario
       must converge, and the voltage error must stay small.

Run:
    python scripts/test_helm_random_vs_pandapower.py
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


N_SCENARIOS = 200
SEED = 0

TIGHT = HelmSolverConfig(n_max=60, epsilon=1e-10)
PAPER = HelmSolverConfig()  # n_max = 30, tol = 1e-5


def build_random_network(rng, config):
    """Return a solved pandapower network for one random scenario."""

    net = build_case5()

    net.ext_grid["vm_pu"] = rng.uniform(0.95, 1.05)
    net.gen["vm_pu"] = rng.uniform(0.95, 1.05, size=len(net.gen))

    scenario = sample_scenario(config, rng)
    apply_demand_uncertainty(net, scenario)

    # apply_wind_output prints one line per bus; keep the log readable.
    with contextlib.redirect_stdout(io.StringIO()):
        apply_wind_output(
            net=net,
            B_w=config.B_w,
            p_w_scale=scenario.p_w_scale,
            p_w_rated_mw=config.p_w_rated_mw,
        )

    pp.runpp(net)

    reference = net.res_bus["vm_pu"].to_numpy() * np.exp(
        1j * np.deg2rad(net.res_bus["va_degree"].to_numpy())
    )

    return net, reference


def main() -> None:

    rng = np.random.default_rng(SEED)
    config = UncertaintyConfig()

    tight_errors = []
    paper_errors = []
    paper_orders = []
    paper_failures = 0

    for _ in range(N_SCENARIOS):

        net, reference = build_random_network(rng, config)
        problem = build_helm_problem_from_network(net)

        tight = solve_helm(problem, TIGHT)
        tight_errors.append(np.max(np.abs(tight.v - reference)))

        paper = solve_helm(problem, PAPER)
        paper_errors.append(np.max(np.abs(paper.v - reference)))
        paper_orders.append(paper.N)

        if not paper.converged:
            paper_failures += 1

    tight_errors = np.asarray(tight_errors)
    paper_errors = np.asarray(paper_errors)
    paper_orders = np.asarray(paper_orders)

    print("=" * 64)
    print(f"HELM vs pandapower, {N_SCENARIOS} random scenarios")
    print("=" * 64)

    print("\nTight tolerance (1e-10):")
    print(f"  max |V_helm - V_pp| : {tight_errors.max():.3e}")

    print("\nPaper tolerance (1e-5, n_max = 30):")
    print(f"  converged           : {N_SCENARIOS - paper_failures}"
          f"/{N_SCENARIOS}")
    print(f"  max |V_helm - V_pp| : {paper_errors.max():.3e}")
    print(f"  terms used          : min {paper_orders.min()}, "
          f"median {int(np.median(paper_orders))}, "
          f"max {paper_orders.max()}")

    assert tight_errors.max() < 1e-7, (
        "HELM does not reproduce the AC power flow."
    )
    assert paper_failures == 0, (
        f"{paper_failures} scenarios did not converge within n_max."
    )
    assert paper_errors.max() < 1e-4, (
        "Paper-tolerance HELM voltage error is unexpectedly large."
    )

    print("\nPASS")


if __name__ == "__main__":
    main()
