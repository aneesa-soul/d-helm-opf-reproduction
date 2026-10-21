"""
Validate the differentiable (PyTorch) D-HELM.

1. Forward, same number of terms:  torch vs NumPy reference solver
   (recursion.solve_helm_series) on a random batch -> ~1e-15.
2. Forward, adaptive stop (Algorithm 1): converged voltages vs
   pandapower, and Q consistency (sum_n Q_n == Im(e) at PV buses).
3. Full action range: how many random samples converge within n_max.
4. Gradient: torch.autograd.gradcheck w.r.t. action and state.
5. Physics: autograd d(output)/d(action) vs central finite differences
   of the pandapower solution.
6. Timing: forward + backward for the paper's 5-bus minibatch (32).

Run:
    python scripts/test_dhelm_differentiable.py
"""

from __future__ import annotations

import contextlib
import io
import time

import numpy as np
import pandapower as pp
import torch

from dhelm_opf.environment.case5 import build_case5
from dhelm_opf.environment.uncertainty import (
    OperatingScenario,
    UncertaintyConfig,
    apply_demand_uncertainty,
    apply_wind_output,
)
from dhelm_opf.helm.differentiable import DHelmNetwork
from dhelm_opf.helm.problem import build_helm_problem_from_network
from dhelm_opf.helm.recursion import solve_helm_series
from dhelm_opf.helm.solver import HelmSolverConfig
from dhelm_opf.models.action_config import action_config_from_network
from dhelm_opf.models.action_decoder import (
    ActionDecoder,
    apply_action_to_network,
)


UNC = UncertaintyConfig()


def random_batch(rng, batch, action_range=0.6):
    """Random states (paper ranges) and normalized actions."""

    n_load = 3
    state = np.concatenate(
        [
            rng.uniform(UNC.demand_min, UNC.demand_max, (batch, 2 * n_load)),
            rng.uniform(UNC.wind_min, UNC.wind_max, (batch, 2)),
        ],
        axis=1,
    )
    action = rng.uniform(-action_range, action_range, (batch, 8))

    return state, action


def pandapower_net(state, action, config, run=True, **runpp_kwargs):
    """Build and (optionally) solve the pandapower net for one sample."""

    net = build_case5()

    scenario = OperatingScenario(
        demand_p_scale=state[0:3],
        demand_q_scale=state[3:6],
        wind_scale=state[6:8],
    )
    apply_demand_uncertainty(net, scenario)

    with contextlib.redirect_stdout(io.StringIO()):
        apply_wind_output(
            net, UNC.wind_buses, scenario.wind_scale, UNC.wind_rated_mw
        )

    apply_action_to_network(net, config, ActionDecoder(config).decode(action))

    if run:
        pp.runpp(net, **runpp_kwargs)

    return net


def as_t(x):
    return torch.as_tensor(x, dtype=torch.float64)


# ----------------------------------------------------------------------
# 1. Same-N comparison with the NumPy reference
# ----------------------------------------------------------------------

def test_vs_numpy(dhelm, config, rng, batch=32):

    state, action = random_batch(rng, batch)
    worst = 0.0

    for n_terms in (1, 2, 4, 8):

        out = dhelm(as_t(state), as_t(action), n_terms=n_terms)
        v_t = out.v_re.detach().numpy() + 1j * out.v_im.detach().numpy()

        for k in range(batch):
            net = pandapower_net(state[k], action[k], config)
            ref = solve_helm_series(
                build_helm_problem_from_network(net),
                max_order=n_terms,
                tol=0.0,              # never stop early: exactly n_terms
            )
            worst = max(worst, np.max(np.abs(v_t[k] - ref["voltage"])))

    assert worst < 1e-12, worst
    print(f"  1. torch vs NumPy (same N)       : max error {worst:.1e}")


# ----------------------------------------------------------------------
# 2. Adaptive stop vs pandapower, Q consistency
# ----------------------------------------------------------------------

def test_vs_pandapower(dhelm, config, rng, batch=64):

    state, action = random_batch(rng, batch)
    out = dhelm(as_t(state), as_t(action))

    assert bool(out.converged.all())

    v_t = out.v_re.detach().numpy() + 1j * out.v_im.detach().numpy()
    worst = 0.0

    for k in range(batch):
        net = pandapower_net(state[k], action[k], config)
        ref = net.res_bus["vm_pu"].to_numpy() * np.exp(
            1j * np.deg2rad(net.res_bus["va_degree"].to_numpy())
        )
        worst = max(worst, np.max(np.abs(v_t[k] - ref)))

    q_gap = torch.max(
        torch.abs(out.q_pv - out.e_im[:, dhelm.pv_idx])
    ).item()

    assert worst < 1e-4, worst
    assert q_gap < 1e-4, q_gap

    print(f"  2. adaptive stop vs pandapower   : N = {out.n_terms}, "
          f"max error {worst:.1e} (tol 1e-5)")
    print(f"     sum Q_n vs Im(e) at PV buses  : max gap {q_gap:.1e}")


# ----------------------------------------------------------------------
# 3. Convergence over the full action range
# ----------------------------------------------------------------------

def test_full_range(dhelm, rng, batch=500):

    state, action = random_batch(rng, batch, action_range=1.0)
    out = dhelm(as_t(state), as_t(action))

    n_conv = int(out.converged.sum())
    print(f"  3. full action range [-1, 1]     : {n_conv}/{batch} "
          f"converged, N = {out.n_terms} "
          f"(n_max = {dhelm.solver_config.max_order}, informational)")


# ----------------------------------------------------------------------
# 4. gradcheck
# ----------------------------------------------------------------------

def test_gradcheck(dhelm, rng):

    state, action = random_batch(rng, 2)

    s = as_t(state).requires_grad_(True)
    a = as_t(action).requires_grad_(True)

    def f(s_, a_):
        out = dhelm(s_, a_, n_terms=6)   # fixed N: constant control flow
        return out.v_re, out.v_im, out.q_pv, out.e_re

    ok = torch.autograd.gradcheck(f, (s, a), eps=1e-6, atol=1e-6)
    assert ok

    print("  4. gradcheck (action and state)  : passed")


# ----------------------------------------------------------------------
# 5. Physics: autograd vs pandapower finite differences
# ----------------------------------------------------------------------

def test_physics_jacobian(config, rng, h=1e-4):
    """
    Outputs compared: |v| at every bus, P injection at the slack bus,
    Q injection at PV + slack buses.  Uses a tight tolerance so the
    truncated series is effectively the exact power-flow solution.
    """

    tight = DHelmNetwork(
        build_case5(),
        config,
        UNC,
        HelmSolverConfig(max_order=60, tolerance=1e-12),
    )

    state, action = random_batch(rng, 1)
    state, action = state[0], action[0]

    slack = tight.slack
    q_buses = sorted(tight.pv + [slack])

    def torch_outputs(a_):
        out = tight(as_t(state), a_)
        return torch.cat(
            [
                out.v_mag[0],
                out.e_re[0, [slack]],
                out.e_im[0, q_buses],
            ]
        )

    jac_auto = torch.autograd.functional.jacobian(
        torch_outputs, as_t(action)
    ).numpy()

    base = config.base_mva

    def pp_outputs(a_):
        net = pandapower_net(state, a_, config, tolerance_mva=1e-11)
        res = net.res_bus
        # pandapower reports bus power in load convention.
        return np.concatenate(
            [
                res["vm_pu"].to_numpy(),
                [-res["p_mw"].to_numpy()[slack] / base],
                -res["q_mvar"].to_numpy()[q_buses] / base,
            ]
        )

    jac_fd = np.zeros_like(jac_auto)
    for j in range(len(action)):
        e = np.zeros_like(action)
        e[j] = h
        jac_fd[:, j] = (pp_outputs(action + e) - pp_outputs(action - e)) / (2 * h)

    rel = np.max(np.abs(jac_auto - jac_fd)) / np.max(np.abs(jac_fd))

    assert rel < 1e-5, rel

    print(f"  5. autograd vs pandapower FD     : relative error {rel:.1e} "
          f"({jac_auto.shape[0]} outputs x {jac_auto.shape[1]} actions)")


# ----------------------------------------------------------------------
# 6. Timing
# ----------------------------------------------------------------------

def test_timing(dhelm, rng, batch=32, repeats=20):

    state, action = random_batch(rng, batch)
    s = as_t(state)
    a = as_t(action).requires_grad_(True)

    # warm-up
    dhelm(s, a).v_mag.sum().backward()

    start = time.perf_counter()
    for _ in range(repeats):
        a.grad = None
        out = dhelm(s, a)
        out.v_mag.sum().backward()
    elapsed = (time.perf_counter() - start) / repeats

    print(f"  6. forward + backward, batch {batch} : "
          f"{1e3 * elapsed:.1f} ms (N = {out.n_terms})")


def main() -> None:

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    net = build_case5()
    config = action_config_from_network(net)
    dhelm = DHelmNetwork(net, config, UNC)

    print("Differentiable D-HELM tests")

    test_vs_numpy(dhelm, config, rng)
    test_vs_pandapower(dhelm, config, rng)
    test_full_range(dhelm, rng)
    test_gradcheck(dhelm, rng)
    test_physics_jacobian(config, rng)
    test_timing(dhelm, rng)

    print("All differentiable D-HELM tests passed.")


if __name__ == "__main__":
    main()
