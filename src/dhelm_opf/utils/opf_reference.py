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
* Ramping (Eq. 3): with a RampConfig, each test episode is solved as a
  sequence of OPFs whose unit limits are the ramp window around the
  OPF's own previous dispatch (see compute_reference).

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
from dhelm_opf.models.action_config import ActionConfig, RampConfig


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


def _slack_row(net: pp.pandapowerNet, fixed_slack_voltage: bool):
    """(table, index) of the slack unit in the OPF network."""

    if fixed_slack_voltage:
        return "ext_grid", int(net.ext_grid.index[0])
    slack_gens = net.gen.index[net.gen["slack"].astype(bool)]
    return "gen", int(slack_gens[0])


def opf_dispatch(net, config: ActionConfig, fixed_slack_voltage: bool):
    """p_g of G_ns (MW) and the slack p_g (MW) of a solved OPF."""

    p_g = np.array([net[f"res_{u.table}"].at[u.index, "p_mw"]
                    for u in config.G_ns])
    table, index = _slack_row(net, fixed_slack_voltage)
    p_g_s = float(net[f"res_{table}"].at[index, "p_mw"])
    return p_g, p_g_s


def ramp_window(config: ActionConfig, ramp: RampConfig, p_g_prev, p_g_s_prev):
    """
    Eq. (3) windows around the previous dispatch (MW):
    G_ns lo/hi arrays and slack lo/hi scalars.
    """

    lo = np.array([max(u.P_min, p - ramp.R_d(u))
                   for u, p in zip(config.G_ns, p_g_prev)])
    hi = np.array([min(u.P_max, p + ramp.R_u(u))
                   for u, p in zip(config.G_ns, p_g_prev)])
    s = config.G_s
    lo_s = max(s.P_min, p_g_s_prev - ramp.R_d(s))
    hi_s = min(s.P_max, p_g_s_prev + ramp.R_u(s))
    return lo, hi, lo_s, hi_s


def opf_as_action(
    net: pp.pandapowerNet,
    config: ActionConfig,
    p_g_min=None,
    p_g_max=None,
) -> np.ndarray:
    """
    The OPF solution expressed as a_hat (Eq. 18 inverted), in the
    config's action order.  p_g_min / p_g_max override the static
    P limits (the Eq. 3 ramp window, when ramping).
    """

    v = np.array([net.res_bus.at[b, "vm_pu"] for b in config.B_vs])
    p = np.array([net[f"res_{u.table}"].at[u.index, "p_mw"] for u in config.G_ns])

    v_lo, v_hi = np.array(config.v_min), np.array(config.v_max)
    p_lo = config.P_min if p_g_min is None else np.asarray(p_g_min)
    p_hi = config.P_max if p_g_max is None else np.asarray(p_g_max)

    a_hat = np.zeros(config.action_dim)
    a_hat[list(config.voltage_action_indices)] = 2 * (v - v_lo) / (v_hi - v_lo) - 1
    a_hat[list(config.generator_action_indices)] = 2 * (p - p_lo) / (p_hi - p_lo) - 1

    # The OPF respects the same bounds; clip only round-off.
    return np.clip(a_hat, -1.0, 1.0)


def compute_reference(
    states: np.ndarray,
    unc: UncertaintyConfig,
    config: ActionConfig,
    fixed_slack_voltage: bool = False,
    ramp: RampConfig | None = None,
    episode_length: int | None = None,
) -> dict:
    """
    Solve the OPF for every state.

    Without ramping each state is independent.  With ramping, states
    are consecutive steps of episodes of `episode_length`, solved as a
    SEQUENCE of OPFs: at t > 0 every unit's P limits (slack included)
    are tightened to the Eq. (3) window around the OPF's own dispatch
    at t - 1.  t = 0 has no previous dispatch and no ramp limit.  If an
    OPF fails, the next step restarts without a ramp limit.
    """

    n = len(states)
    cost = np.full(n, np.nan)
    converged = np.zeros(n, dtype=bool)
    action = np.full((n, config.action_dim), np.nan)
    p_g = np.full((n, config.n_g), np.nan)
    p_g_s = np.full(n, np.nan)
    has_prev = np.zeros(n, dtype=bool)
    p_g_prev = np.full((n, config.n_g), np.nan)
    p_g_s_prev = np.full(n, np.nan)

    L = episode_length if ramp is not None else 1
    if ramp is not None and (L is None or n % L):
        raise ValueError("With ramping, states must be whole episodes.")

    prev = None
    for k, s in enumerate(states):

        if k % L == 0:
            prev = None                      # episode start

        net = opf_network(s, unc, fixed_slack_voltage)
        lo = hi = None

        if ramp is not None and prev is not None:
            lo, hi, lo_s, hi_s = ramp_window(config, ramp, *prev)
            for unit, l, h in zip(config.G_ns, lo, hi):
                net[unit.table].at[unit.index, "min_p_mw"] = l
                net[unit.table].at[unit.index, "max_p_mw"] = h
            table, index = _slack_row(net, fixed_slack_voltage)
            net[table].at[index, "min_p_mw"] = lo_s
            net[table].at[index, "max_p_mw"] = hi_s
            has_prev[k] = True
            p_g_prev[k], p_g_s_prev[k] = prev

        if solve_opf(net):
            converged[k] = True
            cost[k] = float(net.res_cost)
            action[k] = opf_as_action(net, config, lo, hi)
            p_g[k], p_g_s[k] = opf_dispatch(net, config, fixed_slack_voltage)
            prev = (p_g[k], p_g_s[k])
        else:
            prev = None

    return {
        "states": np.asarray(states, dtype=float),
        "cost": cost,
        "converged": converged,
        "action": action,
        "p_g": p_g,
        "p_g_s": p_g_s,
        "has_prev": has_prev,
        "p_g_prev": p_g_prev,
        "p_g_s_prev": p_g_s_prev,
        "fixed_slack_voltage": np.array(fixed_slack_voltage),
        "p_first": np.array(config.p_first),
        "ramp_frac": np.array([ramp.frac_u, ramp.frac_d] if ramp else [np.nan, np.nan]),
        "episode_length": np.array(L if ramp is not None else 0),
        "pandapower_version": np.array(pp.__version__),
    }


def load_or_compute(
    path: str | Path,
    states: np.ndarray,
    unc: UncertaintyConfig,
    config: ActionConfig,
    fixed_slack_voltage: bool = False,
    verbose: bool = True,
    ramp: RampConfig | None = None,
    episode_length: int | None = None,
) -> dict:
    """Load a cached reference for exactly this setup, or compute it."""

    path = Path(path)
    want_ramp = np.array([ramp.frac_u, ramp.frac_d] if ramp else [np.nan, np.nan])
    want_L = episode_length if ramp is not None else 0

    if path.exists():
        data = dict(np.load(path, allow_pickle=False))
        if (
            data["states"].shape == np.shape(states)
            and np.array_equal(data["states"], states)
            and bool(data["fixed_slack_voltage"]) == fixed_slack_voltage
            and "p_first" in data
            and bool(data["p_first"]) == config.p_first
            and np.array_equal(data["ramp_frac"], want_ramp, equal_nan=True)
            and int(data["episode_length"]) == want_L
        ):
            return data
        if verbose:
            print(f"OPF cache {path} does not match this setup; recomputing.")

    if verbose:
        print(f"Solving the OPF for {len(states)} states "
              f"(cached afterwards in {path}) ...")

    data = compute_reference(states, unc, config, fixed_slack_voltage,
                             ramp, episode_length)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    return data
