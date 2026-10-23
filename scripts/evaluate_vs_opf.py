"""
Evaluate a trained D-HELM policy against pandapower's AC OPF.

For each held-out state:
    policy  : action -> D-HELM critic -> cost, violations z
    OPF     : pandapower runopp (interior point, init="pf")
    OPF@critic : the OPF solution converted to a normalized action and
                 passed through the critic.  Its cost must equal the
                 OPF cost and its z must be ~0; this confirms that the
                 OPF solution is feasible for the critic.

Making the OPF solve the same problem as the policy
---------------------------------------------------
* Slack voltage.  pandapower's OPF holds an ext_grid at its voltage
  setpoint, but in the paper the slack voltage is an action (B_s in
  Eq. 4), bounded only by the bus limits.  By default the ext_grid is
  replaced, for the OPF only, by an equivalent slack `gen` (same P/Q
  limits and cost), whose voltage the OPF optimizes within the bus
  limits.  `--fixed-slack-voltage` keeps pandapower's default.
* Unrated lines (max_i_ka = 99999) are left unconstrained, as in the
  critic (see opf_for_state).
* Line limits: pandapower's OPF already uses current magnitude
  (OPF_FLOW_LIM = 2), matching Eqs. (16d)-(16e).

Reported:
    optimality gap  (policy cost - OPF cost) / OPF cost
    feasibility     share of states with total violation <= 1e-6
    timing          policy + critic for the whole set vs OPF per state

Example:
    python scripts/evaluate_vs_opf.py results/runs/case5_L1
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import time
from pathlib import Path

import numpy as np
import pandapower as pp
import torch

from dhelm_opf.algorithms.case5_setup import make_case5_agent
from dhelm_opf.algorithms.dhelm_cpo import CPOConfig
from dhelm_opf.environment.case5 import build_case5
from dhelm_opf.environment.state_sampler import StateSampler
from dhelm_opf.environment.uncertainty import (
    OperatingScenario,
    apply_demand_uncertainty,
    apply_wind_output,
)

FEASIBLE_TOL = 1e-6


def free_slack_voltage(net):
    """
    Replace the ext_grid by an equivalent slack gen, so that the OPF
    optimizes the slack voltage within the bus limits (as the policy
    does).  P/Q limits and cost are copied unchanged.
    """

    eg = net.ext_grid.iloc[0]
    cost = net.poly_cost[net.poly_cost["et"] == "ext_grid"].iloc[0]

    g = pp.create_gen(
        net, bus=int(eg["bus"]), p_mw=0.0, vm_pu=float(eg["vm_pu"]),
        slack=True, controllable=True,
        min_p_mw=eg["min_p_mw"], max_p_mw=eg["max_p_mw"],
        min_q_mvar=eg["min_q_mvar"], max_q_mvar=eg["max_q_mvar"],
    )

    net.poly_cost = net.poly_cost[net.poly_cost["et"] != "ext_grid"]
    pp.create_poly_cost(
        net, g, "gen",
        cp0_eur=cost["cp0_eur"],
        cp1_eur_per_mw=cost["cp1_eur_per_mw"],
        cp2_eur_per_mw2=cost["cp2_eur_per_mw2"],
    )
    net.ext_grid.drop(net.ext_grid.index, inplace=True)


def opf_for_state(state, unc, fixed_slack_voltage=False):
    """Solve the AC OPF for one state.  Returns the solved net or None."""

    net = build_case5()
    scenario = OperatingScenario(state[0:3], state[3:6], state[6:8])
    apply_demand_uncertainty(net, scenario)
    with contextlib.redirect_stdout(io.StringIO()):
        apply_wind_output(net, unc.wind_buses, scenario.wind_scale,
                          unc.wind_rated_mw)

    # Unrated lines (pandapower's 99999 kA placeholder for MATPOWER
    # rateA = 0) are left unconstrained, as in the critic.  Without
    # this the OPF receives ~4e7 MVA flow limits, the interior-point
    # multipliers on those rows blow up (~3e6) and runopp fails on
    # most states with high wind at bus 4 (28/100 converged vs 100/100).
    unrated = net.line["max_i_ka"] >= 1e4
    net.line.loc[unrated, "max_loading_percent"] = np.nan

    if not fixed_slack_voltage:
        free_slack_voltage(net)

    for init in ("pf", "flat"):
        try:
            pp.runopp(net, init=init)
            return net
        except Exception:
            continue
    return None


def opf_as_action(net, config):
    """Convert an OPF solution into the normalized action space."""

    v = [net.res_bus.at[b, "vm_pu"] for b in config.voltage_buses]
    p = [net[f"res_{u.table}"].at[u.index, "p_mw"] for u in config.generators]

    v_lo, v_hi = np.array(config.voltage_min), np.array(config.voltage_max)
    p_lo, p_hi = config.generator_p_min_mw, config.generator_p_max_mw

    a = np.concatenate([
        2 * (np.array(v) - v_lo) / (v_hi - v_lo) - 1,
        2 * (np.array(p) - p_lo) / (p_hi - p_lo) - 1,
    ])
    # The OPF respects the same bounds; clip only round-off.
    return np.clip(a, -1.0, 1.0)


def main() -> None:

    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--seed", type=int, default=20_000)
    p.add_argument("--fixed-slack-voltage", action="store_true",
                   help="pandapower default: hold the slack at its setpoint")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    ckpt = torch.load(run_dir / "policy.pt", weights_only=False)
    run_args = ckpt["args"]
    window = run_args["window"]
    T = run_args["episode_length"]

    agent = make_case5_agent(
        seed=run_args["seed"],
        cpo_config=CPOConfig(**ckpt["cpo_config"]),
    )
    agent.policy.load_state_dict(ckpt["policy"])
    agent.policy.eval()
    agent.cpo.lambdas = ckpt["lambdas"]

    critic, config, unc = agent.critic, agent.action_config, agent.uncertainty

    # ---- held-out episodes, evaluated in the training window length ----
    sampler = StateSampler(agent.dhelm.n_load, agent.dhelm.n_wind, unc,
                           seed=args.seed)
    states = sampler.sample(args.episodes * T)                  # (S, d)
    windows = states.reshape(-1, window, states.shape[1])       # (S/L, L, d)

    t0 = time.perf_counter()
    with torch.no_grad():
        ev = agent.cpo.evaluate(torch.as_tensor(windows, dtype=torch.float32))
    t_policy = time.perf_counter() - t0

    pol_cost = ev.cost.numpy()
    pol_z = ev.z.sum(dim=1).numpy()

    # ---- OPF reference ---------------------------------------------
    opf_cost = np.full(len(states), np.nan)
    opf_crit_cost = np.full(len(states), np.nan)
    opf_crit_z = np.full(len(states), np.nan)

    t0 = time.perf_counter()
    nets = [opf_for_state(s, unc, args.fixed_slack_voltage) for s in states]
    t_opf = time.perf_counter() - t0

    for k, net in enumerate(nets):
        if net is None:
            continue
        opf_cost[k] = net.res_cost
        a = opf_as_action(net, config)
        with torch.no_grad():
            out = critic(torch.as_tensor(states[k:k + 1]),
                         torch.as_tensor(a[None]))
        opf_crit_cost[k] = out.cost.item()
        opf_crit_z[k] = out.z.sum().item()

    ok = ~np.isnan(opf_cost)
    gap = (pol_cost - opf_cost) / opf_cost
    feas = pol_z <= FEASIBLE_TOL

    # ---- report ----------------------------------------------------
    def pct(x):
        return f"{100 * x:+.2f}%"

    print("=" * 66)
    print(f"Policy vs pandapower AC OPF — {run_dir.name}  "
          f"({len(states)} held-out states, window {window})")
    print(f"OPF slack voltage: "
          f"{'fixed at setpoint' if args.fixed_slack_voltage else 'optimized (matches policy)'}")
    print("=" * 66)
    print(f"OPF converged                 : {ok.sum()}/{len(states)}")
    print(f"OPF@critic cost mismatch      : "
          f"{np.nanmax(np.abs(opf_crit_cost - opf_cost)):.2e} $/h (max)")
    print(f"OPF@critic total violation    : {np.nanmax(opf_crit_z):.2e} (max)")
    print()
    print(f"Policy feasible               : {feas.mean():.1%} "
          f"(Σz <= {FEASIBLE_TOL:g})")
    print(f"Policy mean violation Σz      : {pol_z.mean():.2e}")
    print(f"Mean cost   policy / OPF      : {pol_cost[ok].mean():.1f} / "
          f"{opf_cost[ok].mean():.1f} $/h")
    print(f"Optimality gap, all states    : mean {pct(np.mean(gap[ok]))}, "
          f"median {pct(np.median(gap[ok]))}, "
          f"95th pct {pct(np.percentile(gap[ok], 95))}")
    sel = ok & feas
    if sel.any():
        print(f"Optimality gap, feasible only : mean {pct(np.mean(gap[sel]))}, "
              f"median {pct(np.median(gap[sel]))}  ({sel.sum()} states)")
    print()
    print(f"Time  policy+critic, all states : {1e3 * t_policy:.1f} ms")
    print(f"Time  OPF, per state            : {1e3 * t_opf / len(states):.1f} ms")

    # per-constraint violation counts
    names = critic.constraint_names
    counts = (ev.z.numpy() > FEASIBLE_TOL).sum(axis=0)
    worst = [(n, c) for n, c in zip(names, counts) if c > 0]
    if worst:
        print("\nPolicy violations by constraint (states):")
        for n, c in worst:
            print(f"  {n:16s} {c}")

    # ---- save per-state table and summary --------------------------
    suffix = "_fixed_slack" if args.fixed_slack_voltage else ""

    with open(run_dir / f"eval_vs_opf{suffix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "policy_cost", "opf_cost", "gap",
                    "policy_violation", "opf_critic_violation"])
        for k in range(len(states)):
            w.writerow([k, pol_cost[k], opf_cost[k], gap[k], pol_z[k],
                        opf_crit_z[k]])

    summary = dict(
        states=int(len(states)),
        opf_converged=int(ok.sum()),
        feasible=float(feas.mean()),
        gap_mean=float(np.mean(gap[ok])),
        gap_median=float(np.median(gap[ok])),
        gap_feasible_mean=float(np.mean(gap[sel])) if sel.any() else None,
        policy_ms_total=1e3 * t_policy,
        opf_ms_per_state=1e3 * t_opf / len(states),
    )
    summary["fixed_slack_voltage"] = bool(args.fixed_slack_voltage)

    with open(run_dir / f"eval_vs_opf{suffix}.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
