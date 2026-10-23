"""
D-HELM physics-driven critic.

Paper basis
-----------
Sec. 3.2.2, Eq. (17) steps 4-6.  Given the batched D-HELM solution
(voltages v and injections e) the critic computes

    (15)   slack active power      p_g = Re(e_s) + p_d,s - p_w,s
    (16a)  PQ-bus voltage limits   v_min <= |v_i| <= v_max,  i in PQ
    (16b)  slack P limits          max{P_min, p*_g - R_d} <= p_g
                                       <= min{P_max, p*_g + R_u}
    (16c)  generator Q limits      q_min <= q_g = Im(e_i) + q_d,i <= q_max,
                                   i in slack ∪ PV
    (16d)  sending-end current     |i_f| = |Y_f v| I_base <= i_max
    (16e)  receiving-end current   |i_t| = |Y_t v| I_base <= i_max
    (6)    violations              z_n = eta_n (max{0, x - x_max}
                                               + max{0, x_min - x})
    (5)    reward                  r = w0 - w_r * sum_g f_g(p_g)

Everything is a torch operation on the D-HELM output, so r and z are
differentiable with respect to the (normalized) action and the state.

Voltages at PV and slack buses are actions bounded by the tanh scaling
(Eq. 18) and are therefore not constraints here, as in the paper.

The Lagrangian combination of r and z (Eqs. 19-20) belongs to the
training algorithm, not to the critic.

Reproduction choices (see REPRODUCTION_STATUS.md)
-------------------------------------------------
* Q limits are checked per bus, following Eq. (16c) literally: the
  limit of a bus is the sum of the limits of the controllable units on
  it.  Bus 0 of the 5-bus case (gen0 +-30, sgen0 +-127.5) therefore
  has +-157.5 MVAr.
* eta_n = 1 / (x_max - x_min), our reading of Table 1's
  "eta = (Z_max - Z_min)^-1".  For line currents the bounds of
  (16d)-(16e) are -i_max .. i_max, so eta = 1 / (2 i_max).
* w_r = 1 / cost_max, our reading of Table 1's "w_r = cost1max", with
  cost_max = sum over all units (slack included) of f_g(P_max).
* Ramping (16b) is supported but off by default: the case data has no
  ramp rates and the paper's states are sampled independently.
* Lines whose max_i_ka is pandapower's "unlimited" placeholder
  (MATPOWER rateA = 0 -> 99999 kA) carry no current constraint.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandapower as pp
import torch

from dhelm_opf.helm.differentiable import DTYPE, DHelmNetwork, DHelmOutput


@dataclass(frozen=True)
class CriticConfig:
    """Critic settings that are not part of the network data."""

    # Eq. (5), Table 1: w0 = 2.
    w0: float = 2.0

    # Eq. (16b) ramp limits of the slack unit in MW per step.
    # None = no ramp constraint (only P_min / P_max).
    ramp_up_mw: float | None = None
    ramp_down_mw: float | None = None

    # max_i_ka at or above this value is treated as "no limit".
    unlimited_current_ka: float = 1e4


@dataclass
class CriticOutput:
    """
    Physical quantities, constraint violations and reward for a batch.

    Units: voltages p.u., power MW / MVAr, currents kA, cost $/h.
    """

    helm: DHelmOutput

    p_gen_mw: torch.Tensor        # (B, n_gen)   non-slack units (actions)
    p_slack_mw: torch.Tensor      # (B,)         Eq. (15)
    q_bus_mvar: torch.Tensor      # (B, n_vbus)  Eq. (16c), voltage buses
    v_pq: torch.Tensor            # (B, n_pq)    Eq. (16a)
    i_from_ka: torch.Tensor       # (B, n_line)  Eq. (16d), all lines
    i_to_ka: torch.Tensor         # (B, n_line)  Eq. (16e), all lines

    cost: torch.Tensor            # (B,)  sum_g f_g(p_g), $/h
    reward: torch.Tensor          # (B,)  Eq. (5)
    z: torch.Tensor               # (B, n_constraint)  Eq. (6)

    @property
    def total_violation(self) -> torch.Tensor:
        return self.z.sum(dim=1)


def violation(x, lower, upper, eta):
    """Eq. (6): eta * (max{0, x - upper} + max{0, lower - x})."""

    return eta * (torch.relu(x - upper) + torch.relu(lower - x))


class DHELMCritic(torch.nn.Module):
    """
    Physics-driven critic built on a DHelmNetwork.

    Parameters
    ----------
    dhelm:
        The differentiable D-HELM for the same network.
    net:
        The base pandapower network (same one used for `dhelm`).  Used
        once here for line data and bus limits; never during training.
    config:
        CriticConfig.
    """

    def __init__(
        self,
        dhelm: DHelmNetwork,
        net: pp.pandapowerNet,
        config: CriticConfig | None = None,
    ) -> None:

        super().__init__()

        self.dhelm = dhelm
        self.config = config or CriticConfig()

        acfg = dhelm.action_config
        base = dhelm.base_mva
        device = dhelm.LU.device

        def t(x):
            return torch.tensor(np.array(x, dtype=float), dtype=DTYPE,
                                device=device)

        net = copy.deepcopy(net)
        pp.runpp(net)

        # ------------------------------------------------------------
        # (16a) PQ-bus voltage limits
        # ------------------------------------------------------------
        pq = dhelm.pq
        v_lo = net.bus.loc[pq, "min_vm_pu"].to_numpy(dtype=float)
        v_hi = net.bus.loc[pq, "max_vm_pu"].to_numpy(dtype=float)

        self.register_buffer("v_lo", t(v_lo))
        self.register_buffer("v_hi", t(v_hi))

        # ------------------------------------------------------------
        # (16b) slack P limits
        # ------------------------------------------------------------
        slack = acfg.slack
        self.p_slack_min = slack.p_min_mw
        self.p_slack_max = slack.p_max_mw

        # ------------------------------------------------------------
        # (16c) Q limits per voltage bus = sum of unit limits on it
        # ------------------------------------------------------------
        vbus = list(acfg.voltage_buses)
        q_lo = np.zeros(len(vbus))
        q_hi = np.zeros(len(vbus))

        for unit in (*acfg.generators, slack):
            k = vbus.index(unit.bus) if unit.bus in vbus else None
            if k is not None:
                q_lo[k] += unit.q_min_mvar
                q_hi[k] += unit.q_max_mvar

        self.register_buffer("q_lo", t(q_lo))
        self.register_buffer("q_hi", t(q_hi))
        self.register_buffer(
            "vbus_idx",
            torch.as_tensor(vbus, dtype=torch.long, device=device),
        )

        # The Q setpoint of controllable sgens is part of q_spec in the
        # D-HELM, but here it belongs to the unit's free Q (per-bus
        # aggregation), so it is added back.
        q_ctrl = np.zeros(dhelm.n_bus)
        for unit in acfg.generators:
            if unit.table == "sgen":
                row = net.sgen.loc[unit.index]
                q_ctrl[unit.bus] += float(row["q_mvar"]) * float(row["scaling"]) / base
        self.register_buffer("q_ctrl", t(q_ctrl))

        # ------------------------------------------------------------
        # (16d)-(16e) line currents
        # ------------------------------------------------------------
        lookup = net._pd2ppc_lookups["branch"]
        if set(lookup) != {"line"} or lookup["line"] != (0, len(net.line)):
            raise ValueError(
                "Only networks whose branches are all lines (in net.line "
                "order) are supported so far; transformers need their "
                "own Yf/Yt rows and base current."
            )

        ppc = net._ppc["internal"]
        Yf = ppc["Yf"].toarray()
        Yt = ppc["Yt"].toarray()

        self.register_buffer("Yf_g", t(np.real(Yf)))
        self.register_buffer("Yf_b", t(np.imag(Yf)))
        self.register_buffer("Yt_g", t(np.real(Yt)))
        self.register_buffer("Yt_b", t(np.imag(Yt)))

        vn_from = net.bus.loc[net.line["from_bus"], "vn_kv"].to_numpy(dtype=float)
        vn_to = net.bus.loc[net.line["to_bus"], "vn_kv"].to_numpy(dtype=float)

        self.register_buffer("ibase_from", t(base / (np.sqrt(3) * vn_from)))
        self.register_buffer("ibase_to", t(base / (np.sqrt(3) * vn_to)))

        i_max = (
            net.line["max_i_ka"] * net.line["df"] * net.line["parallel"]
        ).to_numpy(dtype=float)

        limited = np.flatnonzero(i_max < self.config.unlimited_current_ka)

        self.register_buffer(
            "line_idx",
            torch.as_tensor(limited, dtype=torch.long, device=device),
        )
        self.register_buffer("i_max", t(i_max[limited]))

        # ------------------------------------------------------------
        # (5) costs: generators in ActionConfig order, then slack
        # ------------------------------------------------------------
        units = list(acfg.generators)
        self.register_buffer("cp0", t([u.cp0 for u in units]))
        self.register_buffer("cp1", t([u.cp1 for u in units]))
        self.register_buffer("cp2", t([u.cp2 for u in units]))
        self.slack_cost = (slack.cp0, slack.cp1, slack.cp2)

        def f(c, p):
            return c[0] + c[1] * p + c[2] * p ** 2

        cost_max = sum(
            f((u.cp0, u.cp1, u.cp2), u.p_max_mw) for u in (*units, slack)
        )
        self.cost_max = float(cost_max)
        self.w_r = 1.0 / self.cost_max

        # ------------------------------------------------------------
        # Constraint names, in the order of z
        # ------------------------------------------------------------
        self.constraint_names = (
            [f"v[bus{b}]" for b in pq]
            + ["p_slack"]
            + [f"q[bus{b}]" for b in vbus]
            + [f"i_from[line{k}]" for k in limited]
            + [f"i_to[line{k}]" for k in limited]
        )

    # ------------------------------------------------------------------

    def _line_currents(self, v_re, v_im):
        """|Y_f v| and |Y_t v| in kA for all lines."""

        def mag(G, B, ibase):
            i_re = v_re @ G.T - v_im @ B.T
            i_im = v_im @ G.T + v_re @ B.T
            return torch.sqrt(i_re ** 2 + i_im ** 2) * ibase

        return (
            mag(self.Yf_g, self.Yf_b, self.ibase_from),
            mag(self.Yt_g, self.Yt_b, self.ibase_to),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        p_slack_prev_mw: torch.Tensor | None = None,
        n_terms: int | None = None,
    ) -> CriticOutput:
        """
        Evaluate the critic.

        p_slack_prev_mw:
            (B,) slack output of the previous step, p*_g in (16b).
            Only used when ramp limits are configured.
        n_terms:
            Passed to the D-HELM (fixed number of terms; for tests).
        """

        cfg = self.config
        base = self.dhelm.base_mva
        s = self.dhelm.slack

        out = self.dhelm(state, action, n_terms=n_terms)
        specs = out.specs

        # ---- (15) slack P: injection minus everything else on the bus
        p_slack = base * (out.e_re[:, s] - specs.p_spec[:, s])

        # ---- (16c) unit Q per voltage bus
        vb = self.vbus_idx
        q_bus = base * (
            out.e_im[:, vb] - specs.q_spec[:, vb] + self.q_ctrl[vb]
        )

        # ---- (16a) PQ voltages
        v_pq = out.v_mag[:, self.dhelm.pq_idx]

        # ---- (16d)-(16e) line currents
        i_from, i_to = self._line_currents(out.v_re, out.v_im)

        # ---- (5) cost and reward
        p_gen = base * specs.p_gen
        c0, c1, c2 = self.slack_cost
        cost = (
            (self.cp0 + self.cp1 * p_gen + self.cp2 * p_gen ** 2).sum(dim=1)
            + c0 + c1 * p_slack + c2 * p_slack ** 2
        )
        reward = cfg.w0 - self.w_r * cost

        # ---- (6) violations
        p_lo = torch.full_like(p_slack, self.p_slack_min)
        p_hi = torch.full_like(p_slack, self.p_slack_max)

        if p_slack_prev_mw is not None:
            prev = torch.as_tensor(p_slack_prev_mw, dtype=DTYPE,
                                   device=p_slack.device).reshape(-1)
            if cfg.ramp_down_mw is not None:
                p_lo = torch.maximum(p_lo, prev - cfg.ramp_down_mw)
            if cfg.ramp_up_mw is not None:
                p_hi = torch.minimum(p_hi, prev + cfg.ramp_up_mw)

        eta_p = 1.0 / (self.p_slack_max - self.p_slack_min)

        i_max = self.i_max
        li = self.line_idx

        z = torch.cat(
            [
                violation(v_pq, self.v_lo, self.v_hi,
                          1.0 / (self.v_hi - self.v_lo)),
                violation(p_slack, p_lo, p_hi, eta_p).unsqueeze(1),
                violation(q_bus, self.q_lo, self.q_hi,
                          1.0 / (self.q_hi - self.q_lo)),
                violation(i_from[:, li], -i_max, i_max, 0.5 / i_max),
                violation(i_to[:, li], -i_max, i_max, 0.5 / i_max),
            ],
            dim=1,
        )

        return CriticOutput(
            helm=out,
            p_gen_mw=p_gen,
            p_slack_mw=p_slack,
            q_bus_mvar=q_bus,
            v_pq=v_pq,
            i_from_ka=i_from,
            i_to_ka=i_to,
            cost=cost,
            reward=reward,
            z=z,
        )

    def describe(self) -> str:
        """Constraint list with bounds and eta."""

        lines = [
            "D-HELM critic constraints (Eq. 6)",
            f"  reward: r = {self.config.w0} - cost / {self.cost_max:.1f}",
        ]
        bounds = (
            [(lo, hi) for lo, hi in zip(self.v_lo.tolist(), self.v_hi.tolist())]
            + [(self.p_slack_min, self.p_slack_max)]
            + [(lo, hi) for lo, hi in zip(self.q_lo.tolist(), self.q_hi.tolist())]
            + [(-i, i) for i in self.i_max.tolist()] * 2
        )
        for name, (lo, hi) in zip(self.constraint_names, bounds):
            lines.append(
                f"  {name:16s} [{lo:9.4f}, {hi:9.4f}]  eta = {1 / (hi - lo):.4g}"
            )
        return "\n".join(lines)
