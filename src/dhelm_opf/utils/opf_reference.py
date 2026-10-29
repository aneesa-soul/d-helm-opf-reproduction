"""
pandapower AC OPF reference, matched to the paper's problem, with a cache.

Matching the OPF to the problem the policy solves
-------------------------------------------------
* Slack voltage.  pandapower's OPF holds an ext_grid at its voltage
  setpoint, but in the paper the slack voltage is an action (B_s in
  Eq. 4) bounded only by the bus limits.  By default the ext_grid is
  replaced, for the OPF only, by an equivalent slack `gen` (same P/Q
  limits and cost) whose voltage the OPF optimizes.
* Unrated lines (max_i_ka = 99999, MATPOWER rateA = 0) are left
  unconstrained, as in the critic.  Otherwise the OPF receives ~4e7 MVA
  limits, its interior-point multipliers blow up and runopp fails on
  most high-wind states (28/100 converged vs 100/100).
* Line limits: pandapower's OPF uses current magnitude
  (OPF_FLOW_LIM = 2), matching Eqs. (16d)-(16e).

Cache
-----
Solving the OPF takes 0.3-0.8 s per state, so the reference for a test
set is computed once and stored as .npz (states, cost, converged,
normalized OPF action).  `load_or_compute()` reuses a cache only if its
states match exactly.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pandapower as pp

from dhelm_opf.environment.case5 import build_case5
from dhelm_opf.environment.uncertainty import (
    OperatingScenario,
    UncertaintyConfig,
    apply_demand_uncertainty,
    apply_wind_output,
)
from dhelm_opf.models.action_config import ActionConfig


def free_slack_voltage(net: pp.pandapowerNet) -> None:
    """
    Replace the ext_grid by an equivalent slack gen so that the OPF
    optimizes the slack voltage within the bus limits.  P/Q limits and
    cost are copied unchanged.
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


def opf_network(
    state: np.ndarray,
    unc: UncertaintyConfig,
    fixed_slack_voltage: bool = False,
) -> pp.pandapowerNet:
    """Case5 with the state's demand and wind, prepared for the OPF."""

    net = build_case5()
    n_d = len(net.load)

    scenario = OperatingScenario(
        state[0:n_d], state[n_d:2 * n_d], state[2 * n_d:]
    )
    apply_demand_uncertainty(net, scenario)
    with contextlib.redirect_stdout(io.StringIO()):
        apply_wind_output(net, unc.B_w, scenario.p_w_scale,
                          unc.p_w_rated_mw)

    unrated = net.line["max_i_ka"] >= 1e4
    net.line.loc[unrated, "max_loading_percent"] = np.nan

    if not fixed_slack_voltage:
        free_slack_voltage(net)

    return net


def solve_opf(net: pp.pandapowerNet) -> bool:
    """runopp from a power-flow start, then a flat start.  True if solved."""

    for init in ("pf", "flat"):
        try:
            pp.runopp(net, init=init)
            return True
        except Exception:
            continue
    return False


def opf_as_action(net: pp.pandapowerNet, config: ActionConfig) -> np.ndarray:
    """The OPF solution expressed in the normalized action space."""

    v = [net.res_bus.at[b, "vm_pu"] for b in config.B_vs]
    p = [net[f"res_{u.table}"].at[u.index, "p_mw"] for u in config.G_ns]

    v_lo, v_hi = np.array(config.v_min), np.array(config.v_max)
    p_lo, p_hi = config.P_min, config.P_max

    a = np.concatenate([
        2 * (np.array(v) - v_lo) / (v_hi - v_lo) - 1,
        2 * (np.array(p) - p_lo) / (p_hi - p_lo) - 1,
    ])
    # The OPF respects the same bounds; clip only round-off.
    return np.clip(a, -1.0, 1.0)


def compute_reference(
    states: np.ndarray,
    unc: UncertaintyConfig,
    config: ActionConfig,
    fixed_slack_voltage: bool = False,
) -> dict:
    """Solve the OPF for every state."""

    n = len(states)
    cost = np.full(n, np.nan)
    converged = np.zeros(n, dtype=bool)
    action = np.full((n, config.action_dim), np.nan)

    for k, s in enumerate(states):
        net = opf_network(s, unc, fixed_slack_voltage)
        if solve_opf(net):
            converged[k] = True
            cost[k] = float(net.res_cost)
            action[k] = opf_as_action(net, config)

    return {
        "states": np.asarray(states, dtype=float),
        "cost": cost,
        "converged": converged,
        "action": action,
        "fixed_slack_voltage": np.array(fixed_slack_voltage),
        "pandapower_version": np.array(pp.__version__),
    }


def load_or_compute(
    path: str | Path,
    states: np.ndarray,
    unc: UncertaintyConfig,
    config: ActionConfig,
    fixed_slack_voltage: bool = False,
    verbose: bool = True,
) -> dict:
    """Load a cached reference for exactly these states, or compute it."""

    path = Path(path)

    if path.exists():
        data = dict(np.load(path, allow_pickle=False))
        if (
            data["states"].shape == np.shape(states)
            and np.array_equal(data["states"], states)
            and bool(data["fixed_slack_voltage"]) == fixed_slack_voltage
        ):
            return data
        if verbose:
            print(f"OPF cache {path} does not match these states; recomputing.")

    if verbose:
        print(f"Solving the OPF for {len(states)} states "
              f"(cached afterwards in {path}) ...")

    data = compute_reference(states, unc, config, fixed_slack_voltage)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    return data
