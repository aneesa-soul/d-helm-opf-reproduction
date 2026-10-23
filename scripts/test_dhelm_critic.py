"""
Validate the D-HELM critic (Eqs. 15, 16, 6, 5).

1. Physical quantities vs pandapower's res_* tables:
   slack P, unit Q per voltage bus, PQ voltages, line currents, cost.
2. Violations z (Eq. 6) and reward (Eq. 5) vs an independent NumPy
   computation that reads the limits straight from the network tables.
   Random actions over the full range, so many constraints are violated.
3. Ramping (16b) with a configured ramp rate.
4. gradcheck of reward and z w.r.t. action and state.

Run:
    python scripts/test_dhelm_critic.py
"""

from __future__ import annotations

import contextlib
import io

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
from dhelm_opf.helm.solver import HelmSolverConfig
from dhelm_opf.models.action_config import action_config_from_network
from dhelm_opf.models.action_decoder import (
    ActionDecoder,
    apply_action_to_network,
)
from dhelm_opf.models.dhelm_critic import CriticConfig, DHELMCritic


UNC = UncertaintyConfig()


def as_t(x):
    return torch.as_tensor(x, dtype=torch.float64)


def random_batch(rng, batch, action_range=1.0):
    state = np.concatenate(
        [
            rng.uniform(UNC.demand_min, UNC.demand_max, (batch, 6)),
            rng.uniform(UNC.wind_min, UNC.wind_max, (batch, 2)),
        ],
        axis=1,
    )
    action = rng.uniform(-action_range, action_range, (batch, 8))
    return state, action


def solved_net(state, action, config):
    net = build_case5()
    scenario = OperatingScenario(state[0:3], state[3:6], state[6:8])
    apply_demand_uncertainty(net, scenario)
    with contextlib.redirect_stdout(io.StringIO()):
        apply_wind_output(net, UNC.wind_buses, scenario.wind_scale,
                          UNC.wind_rated_mw)
    apply_action_to_network(net, config, ActionDecoder(config).decode(action))
    pp.runpp(net, tolerance_mva=1e-10)
    return net


# ----------------------------------------------------------------------
# Independent NumPy reference built from pandapower results
# ----------------------------------------------------------------------

def eq6(x, lo, hi):
    """Eq. (6) with eta = 1 / (hi - lo)."""
    return (np.maximum(0, x - hi) + np.maximum(0, lo - x)) / (hi - lo)


def reference(net, voltage_buses):
    """Quantities, z and reward computed from pandapower and net tables."""

    res = net.res_bus
    eg = net.ext_grid.iloc[0]
    slack_bus = int(eg["bus"])

    pq = [b for b in net.bus.index
          if b not in voltage_buses]

    # unit Q per voltage bus (controllable units only, as in Eq. 16c)
    q_bus, q_lo, q_hi = [], [], []
    for b in voltage_buses:
        q = lo = hi = 0.0
        for i, row in net.gen.iterrows():
            if row["bus"] == b:
                q += net.res_gen.at[i, "q_mvar"]
                lo += row["min_q_mvar"]
                hi += row["max_q_mvar"]
        for i, row in net.sgen.iterrows():
            if row["bus"] == b and row["controllable"] and row["type"] != "WP":
                q += net.res_sgen.at[i, "q_mvar"]
                lo += row["min_q_mvar"]
                hi += row["max_q_mvar"]
        if b == slack_bus:
            q += net.res_ext_grid.at[0, "q_mvar"]
            lo += eg["min_q_mvar"]
            hi += eg["max_q_mvar"]
        q_bus.append(q)
        q_lo.append(lo)
        q_hi.append(hi)
    q_bus, q_lo, q_hi = map(np.array, (q_bus, q_lo, q_hi))

    p_slack = net.res_ext_grid.at[0, "p_mw"]
    v_pq = res.loc[pq, "vm_pu"].to_numpy()

    i_max = net.line["max_i_ka"].to_numpy()
    limited = i_max < 1e4
    i_from = net.res_line["i_from_ka"].to_numpy()
    i_to = net.res_line["i_to_ka"].to_numpy()

    # cost from poly_cost and pandapower's dispatch
    cost = 0.0
    cost_max = 0.0
    for _, c in net.poly_cost.iterrows():
        table, idx = c["et"], int(c["element"])
        p = net[f"res_{table}"].at[idx, "p_mw"]
        pmax = net[table].at[idx, "max_p_mw"]
        coef = (c["cp0_eur"], c["cp1_eur_per_mw"], c["cp2_eur_per_mw2"])
        cost += coef[0] + coef[1] * p + coef[2] * p ** 2
        cost_max += coef[0] + coef[1] * pmax + coef[2] * pmax ** 2

    z = np.concatenate(
        [
            eq6(v_pq, net.bus.loc[pq, "min_vm_pu"].to_numpy(),
                net.bus.loc[pq, "max_vm_pu"].to_numpy()),
            [eq6(p_slack, eg["min_p_mw"], eg["max_p_mw"])],
            eq6(q_bus, q_lo, q_hi),
            eq6(i_from[limited], -i_max[limited], i_max[limited]),
            eq6(i_to[limited], -i_max[limited], i_max[limited]),
        ]
    )

    return dict(
        p_slack=p_slack, q_bus=q_bus, v_pq=v_pq,
        i_from=i_from, i_to=i_to,
        cost=cost, cost_max=cost_max,
        reward=2.0 - cost / cost_max, z=z,
    )


# ----------------------------------------------------------------------

def test_vs_reference(critic, config, rng, batch=100):

    state, action = random_batch(rng, batch)
    out = critic(as_t(state), as_t(action))

    assert bool(out.helm.converged.all())

    err = {k: 0.0 for k in
           ("p_slack", "q_bus", "v_pq", "i_from", "i_to", "cost",
            "reward", "z")}
    violated = np.zeros(len(critic.constraint_names), dtype=int)
    cost_max = None

    for k in range(batch):
        ref = reference(solved_net(state[k], action[k], config),
                        list(config.voltage_buses))
        cost_max = ref["cost_max"]

        mine = dict(
            p_slack=out.p_slack_mw[k].item(),
            q_bus=out.q_bus_mvar[k].detach().numpy(),
            v_pq=out.v_pq[k].detach().numpy(),
            i_from=out.i_from_ka[k].detach().numpy(),
            i_to=out.i_to_ka[k].detach().numpy(),
            cost=out.cost[k].item(),
            reward=out.reward[k].item(),
            z=out.z[k].detach().numpy(),
        )
        for key in err:
            err[key] = max(err[key],
                           float(np.max(np.abs(np.asarray(mine[key])
                                               - np.asarray(ref[key])))))
        violated += ref["z"] > 0

    assert abs(cost_max - critic.cost_max) < 1e-9, cost_max

    tol = dict(p_slack=1e-5, q_bus=1e-5, v_pq=1e-8, i_from=1e-8,
               i_to=1e-8, cost=1e-3, reward=1e-8, z=1e-7)

    print(f"  1-2. critic vs pandapower + NumPy ({batch} random samples)")
    for key, value in err.items():
        print(f"       {key:8s} max abs error {value:.1e}")
        assert value < tol[key], (key, value)

    print("       samples violating each constraint:")
    for name, count in zip(critic.constraint_names, violated):
        print(f"         {name:16s} {count:3d}/{batch}")


def test_ramping(dhelm, net, rng):

    critic = DHELMCritic(
        dhelm, net, CriticConfig(ramp_up_mw=10.0, ramp_down_mw=10.0)
    )

    # Find a sample whose slack output is well inside [0, 200] MW.
    for _ in range(200):
        state, action = random_batch(rng, 1, action_range=0.6)
        out = critic(as_t(state), as_t(action))
        p = out.p_slack_mw.item()
        if 30.0 < p < 170.0:
            break
    else:
        raise RuntimeError("No interior slack sample found.")

    k = critic.constraint_names.index("p_slack")
    eta = 1.0 / 200.0

    up = critic(as_t(state), as_t(action), p_slack_prev_mw=as_t([p - 25.0]))
    down = critic(as_t(state), as_t(action), p_slack_prev_mw=as_t([p + 25.0]))
    inside = critic(as_t(state), as_t(action), p_slack_prev_mw=as_t([p - 5.0]))
    no_prev = critic(as_t(state), as_t(action))

    assert np.isclose(up.z[0, k].item(), 15.0 * eta)
    assert np.isclose(down.z[0, k].item(), 15.0 * eta)
    assert inside.z[0, k].item() == 0.0
    assert no_prev.z[0, k].item() == 0.0

    print(f"  3. ramping (16b), +-10 MW/step   : OK "
          f"(slack {p:.1f} MW, 15 MW over -> z = {up.z[0, k].item():.4f})")


def test_gradcheck(critic, rng):

    state, action = random_batch(rng, 3)

    s = as_t(state).requires_grad_(True)
    a = as_t(action).requires_grad_(True)

    def f(s_, a_):
        out = critic(s_, a_, n_terms=8)
        return out.reward, out.z

    assert torch.autograd.gradcheck(f, (s, a), eps=1e-6, atol=1e-6)

    out = critic(as_t(state), as_t(action), n_terms=8)
    n_active = int((out.z > 0).sum())
    print(f"  4. gradcheck (reward, z)         : passed "
          f"({n_active} active violations in the batch)")


def main() -> None:

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    net = build_case5()
    config = action_config_from_network(net)

    dhelm = DHelmNetwork(net, config, UNC,
                         HelmSolverConfig(max_order=60, tolerance=1e-10))
    critic = DHELMCritic(dhelm, net)

    print(critic.describe())
    print()
    print("D-HELM critic tests")

    test_vs_reference(critic, config, rng)
    test_ramping(dhelm, net, rng)
    test_gradcheck(critic, rng)

    print("All D-HELM critic tests passed.")


if __name__ == "__main__":
    main()
