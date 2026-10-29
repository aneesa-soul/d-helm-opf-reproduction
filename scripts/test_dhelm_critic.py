"""
Validate the D-HELM critic (Eqs. 15, 16, 6, 5).

1. Physical quantities vs pandapower's res_* tables:
   slack P, unit Q per voltage bus, PQ voltages, line currents, cost.
2. Violations z (Eq. 6) and reward (Eq. 5) vs an independent NumPy
   computation that reads the limits straight from the network tables.
   Random actions over the full range, so many constraints are violated.
3. Ramping (16b) with a configured ramp rate (RampConfig, 5% of
   the 200 MW slack = 10 MW/step).
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
from dhelm_opf.models.action_config import RampConfig, action_config_from_network
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
        apply_wind_output(net, UNC.B_w, scenario.p_w_scale,
                          UNC.p_w_rated_mw)
    apply_action_to_network(net, config, ActionDecoder(config).decode(action))
    pp.runpp(net, tolerance_mva=1e-10)
    return net


# ----------------------------------------------------------------------
# Independent NumPy reference built from pandapower results
# ----------------------------------------------------------------------

def eq6(x, lo, hi):
    """Eq. (6) with eta = 1 / (hi - lo)."""
    return (np.maximum(0, x - hi) + np.maximum(0, lo - x)) / (hi - lo)


def reference(net, B_vs):
    """Quantities, z and r computed from pandapower and net tables."""

    res = net.res_bus
    eg = net.ext_grid.iloc[0]
    B_s = int(eg["bus"])

    B_q = [b for b in net.bus.index
          if b not in B_vs]

    # unit Q per voltage bus (controllable units only, as in Eq. 16c)
    q_g, q_g_min, q_g_max = [], [], []
    for b in B_vs:
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
        if b == B_s:
            q += net.res_ext_grid.at[0, "q_mvar"]
            lo += eg["min_q_mvar"]
            hi += eg["max_q_mvar"]
        q_g.append(q)
        q_g_min.append(lo)
        q_g_max.append(hi)
    q_g, q_g_min, q_g_max = map(np.array, (q_g, q_g_min, q_g_max))

    p_g_s = net.res_ext_grid.at[0, "p_mw"]
    v_q = res.loc[B_q, "vm_pu"].to_numpy()

    i_max = net.line["max_i_ka"].to_numpy()
    limited = i_max < 1e4
    i_f = net.res_line["i_from_ka"].to_numpy()
    i_t = net.res_line["i_to_ka"].to_numpy()

    # f_sum from poly_cost and pandapower's dispatch
    f_sum = 0.0
    f_max = 0.0
    for _, c in net.poly_cost.iterrows():
        table, idx = c["et"], int(c["element"])
        p = net[f"res_{table}"].at[idx, "p_mw"]
        pmax = net[table].at[idx, "max_p_mw"]
        coef = (c["cp0_eur"], c["cp1_eur_per_mw"], c["cp2_eur_per_mw2"])
        f_sum += coef[0] + coef[1] * p + coef[2] * p ** 2
        f_max += coef[0] + coef[1] * pmax + coef[2] * pmax ** 2

    z = np.concatenate(
        [
            eq6(v_q, net.bus.loc[B_q, "min_vm_pu"].to_numpy(),
                net.bus.loc[B_q, "max_vm_pu"].to_numpy()),
            [eq6(p_g_s, eg["min_p_mw"], eg["max_p_mw"])],
            eq6(q_g, q_g_min, q_g_max),
            eq6(i_f[limited], -i_max[limited], i_max[limited]),
            eq6(i_t[limited], -i_max[limited], i_max[limited]),
        ]
    )

    return dict(
        p_g_s=p_g_s, q_g=q_g, v_q=v_q,
        i_f=i_f, i_t=i_t,
        f_sum=f_sum, f_max=f_max,
        r=2.0 - f_sum / f_max, z=z,
    )


# ----------------------------------------------------------------------

def test_vs_reference(critic, config, rng, batch=100):

    state, action = random_batch(rng, batch)
    out = critic(as_t(state), as_t(action))

    assert bool(out.helm.converged.all())

    err = {k: 0.0 for k in
           ("p_g_s", "q_g", "v_q", "i_f", "i_t", "f_sum",
            "r", "z")}
    violated = np.zeros(len(critic.constraint_names), dtype=int)
    f_max = None

    for k in range(batch):
        ref = reference(solved_net(state[k], action[k], config),
                        list(config.B_vs))
        f_max = ref["f_max"]

        mine = dict(
            p_g_s=out.p_g_s[k].item(),
            q_g=out.q_g[k].detach().numpy(),
            v_q=out.v_q[k].detach().numpy(),
            i_f=out.i_f[k].detach().numpy(),
            i_t=out.i_t[k].detach().numpy(),
            f_sum=out.f_sum[k].item(),
            r=out.r[k].item(),
            z=out.z[k].detach().numpy(),
        )
        for key in err:
            err[key] = max(err[key],
                           float(np.max(np.abs(np.asarray(mine[key])
                                               - np.asarray(ref[key])))))
        violated += ref["z"] > 0

    assert abs(f_max - critic.f_max) < 1e-9, f_max

    tol = dict(p_g_s=1e-5, q_g=1e-5, v_q=1e-8, i_f=1e-8,
               i_t=1e-8, f_sum=1e-3, r=1e-8, z=1e-7)

    print(f"  1-2. critic vs pandapower + NumPy ({batch} random samples)")
    for key, value in err.items():
        print(f"       {key:8s} max abs error {value:.1e}")
        assert value < tol[key], (key, value)

    print("       samples violating each constraint:")
    for name, count in zip(critic.constraint_names, violated):
        print(f"         {name:16s} {count:3d}/{batch}")


def test_ramping(dhelm, net, rng):

    ramped = DHelmNetwork(net, dhelm.action_config, UNC,
                          HelmSolverConfig(n_max=60, epsilon=1e-10),
                          ramp=RampConfig(frac_u=0.05, frac_d=0.05))
    critic = DHELMCritic(ramped, net)
    assert np.isclose(ramped.R_u_s, 10.0) and np.isclose(ramped.R_d_s, 10.0)

    # Find a sample whose slack output is well inside [0, 200] MW.
    for _ in range(200):
        state, action = random_batch(rng, 1, action_range=0.6)
        out = critic(as_t(state), as_t(action))
        p = out.p_g_s.item()
        if 30.0 < p < 170.0:
            break
    else:
        raise RuntimeError("No interior slack sample found.")

    k = critic.constraint_names.index("p_g[slack]")
    eta = 1.0 / 200.0

    up = critic(as_t(state), as_t(action), p_g_s_prev=as_t([p - 25.0]))
    down = critic(as_t(state), as_t(action), p_g_s_prev=as_t([p + 25.0]))
    inside = critic(as_t(state), as_t(action), p_g_s_prev=as_t([p - 5.0]))
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
        out = critic(s_, a_, N=8)
        return out.r, out.z

    assert torch.autograd.gradcheck(f, (s, a), eps=1e-6, atol=1e-6)

    out = critic(as_t(state), as_t(action), N=8)
    n_active = int((out.z > 0).sum())
    print(f"  4. gradcheck (reward, z)         : passed "
          f"({n_active} active violations in the batch)")


def main() -> None:

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    net = build_case5()
    config = action_config_from_network(net)

    dhelm = DHelmNetwork(net, config, UNC,
                         HelmSolverConfig(n_max=60, epsilon=1e-10))
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
