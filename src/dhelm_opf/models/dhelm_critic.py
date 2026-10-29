"""
D-HELM physics-driven critic.

Paper basis
-----------
Sec. 3.2.2, Eq. (17) steps 4-6.  Given the batched D-HELM solution
(voltages v and injections e) the critic computes

    (15)   slack active power      p_g = Re(e_i) - p_i^w + p_i^d,  i in B_s
    (16a)  PQ-bus voltage limits   v_min <= |v_i| <= v_max,        i in B_q
    (16b)  slack P limits          max{P_min, p*_g - R_d} <= p_g
                                       <= min{P_max, R_u + p*_g}
    (16c)  unit Q limits           q_g_min <= q_g = Im(e_i) + q_i^d <= q_g_max,
                                   i in B_s ∪ B_v
    (16d)  sending-end current     -i_max <= i_f = Y_f v I_base <= i_max
    (16e)  receiving-end current   -i_max <= i_t = Y_t v I_base <= i_max
    (6)    violations              z_n = eta_n (max{0, x_n - x_max}
                                                + max{0, x_min - x_n})
    (5)    reward                  r = w_0 - w_r * sum_g f_g(p_g)

Everything is a torch operation on the D-HELM output, so r and z are
differentiable with respect to a_hat and s.

Voltages at PV and slack buses are actions bounded by the tanh scaling
(Eq. 18) and are therefore not constraints here, as in the paper.

The Lagrangian combination of r and z (Eqs. 19-20) belongs to the
training algorithm, not to the critic.

Notation follows the paper (see NOTATION.md).

Reproduction choices (see REPRODUCTION_STATUS.md)
-------------------------------------------------
* Q limits are checked per bus, following Eq. (16c) literally: the
  limit of a bus is the sum of the limits of the controllable units on
  it.  Bus 0 of the 5-bus case (gen0 +-30, sgen0 +-127.5) therefore
  has +-157.5 MVAr.
* eta_n = 1 / (x_max - x_min), our reading of Table 1's
  "eta = (Z_max - Z_min)^-1".  For line currents the bounds of
  (16d)-(16e) are -i_max .. i_max, so eta = 1 / (2 i_max).
* w_r = 1 / f_max, our reading of Table 1's "w_r = cost1max", with
  f_max = sum over all units (slack included) of f_g(P_max).
* Ramping: R_u, R_d come from the D-HELM's RampConfig (Eq. 3 rates are
  not given in the paper).  The slack ramp enters z through (16b); the
  non-slack units are kept inside their ramp window by the Eq. (18)
  scaling in the D-HELM.
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

    # Eq. (5), Table 1: w_0 = 2.
    w_0: float = 2.0

    # max_i_ka at or above this value is treated as "no limit".
    unlimited_current_ka: float = 1e4


@dataclass
class CriticOutput:
    """
    Physical quantities, constraint violations and reward for a batch.

    Units: |v| p.u., power MW / MVAr, currents kA, cost $/h.
    """

    helm: DHelmOutput

    p_g: torch.Tensor          # (B, n_g)      p_g of G_ns (actions), MW
    p_g_s: torch.Tensor        # (B,)          slack p_g, Eq. (15), MW
    q_g: torch.Tensor          # (B, |B_vs|)   unit Q per bus, Eq. (16c), MVAr
    v_q: torch.Tensor          # (B, |B_q|)    |v_i|, i in B_q, Eq. (16a)
    i_f: torch.Tensor          # (B, n_line)   |i_f|, Eq. (16d), kA
    i_t: torch.Tensor          # (B, n_line)   |i_t|, Eq. (16e), kA

    f_sum: torch.Tensor        # (B,)  sum_g f_g(p_g), $/h
    r: torch.Tensor            # (B,)  Eq. (5)
    z: torch.Tensor            # (B, n_constraint)  Eq. (6)

    @property
    def z_sum(self) -> torch.Tensor:
        return self.z.sum(dim=1)


def violation(x, x_min, x_max, eta):
    """Eq. (6): eta * (max{0, x - x_max} + max{0, x_min - x})."""

    return eta * (torch.relu(x - x_max) + torch.relu(x_min - x))


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
        device = dhelm.A_LU.device

        def t(x):
            return torch.tensor(np.array(x, dtype=float), dtype=DTYPE,
                                device=device)

        net = copy.deepcopy(net)
        pp.runpp(net)

        # ------------------------------------------------------------
        # (16a) PQ-bus voltage limits v_min, v_max
        # ------------------------------------------------------------
        B_q = dhelm.B_q
        v_min = net.bus.loc[B_q, "min_vm_pu"].to_numpy(dtype=float)
        v_max = net.bus.loc[B_q, "max_vm_pu"].to_numpy(dtype=float)

        self.register_buffer("v_min", t(v_min))
        self.register_buffer("v_max", t(v_max))

        # ------------------------------------------------------------
        # (16b) slack P limits
        # ------------------------------------------------------------
        G_s = acfg.G_s
        self.P_min_s = G_s.P_min
        self.P_max_s = G_s.P_max

        # ------------------------------------------------------------
        # (16c) Q limits per voltage bus = sum of unit limits on it
        # ------------------------------------------------------------
        B_vs = list(acfg.B_vs)
        q_g_min = np.zeros(len(B_vs))
        q_g_max = np.zeros(len(B_vs))

        for unit in (*acfg.G_ns, G_s):
            k = B_vs.index(unit.bus) if unit.bus in B_vs else None
            if k is not None:
                q_g_min[k] += unit.q_min
                q_g_max[k] += unit.q_max

        self.register_buffer("q_g_min", t(q_g_min))
        self.register_buffer("q_g_max", t(q_g_max))
        self.register_buffer(
            "B_vs_idx",
            torch.as_tensor(B_vs, dtype=torch.long, device=device),
        )

        # The Q setpoint of controllable sgens is part of q (Eq. 8) in
        # the D-HELM, but here it belongs to the unit's free q_g
        # (per-bus aggregation), so it is added back.
        q_ctrl = np.zeros(dhelm.n_bus)
        for unit in acfg.G_ns:
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
                "own Y_f/Y_t rows and base current."
            )

        ppc = net._ppc["internal"]
        Y_f = ppc["Yf"].toarray()
        Y_t = ppc["Yt"].toarray()

        self.register_buffer("Y_f_re", t(np.real(Y_f)))
        self.register_buffer("Y_f_im", t(np.imag(Y_f)))
        self.register_buffer("Y_t_re", t(np.real(Y_t)))
        self.register_buffer("Y_t_im", t(np.imag(Y_t)))

        vn_from = net.bus.loc[net.line["from_bus"], "vn_kv"].to_numpy(dtype=float)
        vn_to = net.bus.loc[net.line["to_bus"], "vn_kv"].to_numpy(dtype=float)

        self.register_buffer("I_base_f", t(base / (np.sqrt(3) * vn_from)))
        self.register_buffer("I_base_t", t(base / (np.sqrt(3) * vn_to)))

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
        # (5) costs f_g: units G_ns in ActionConfig order, then G_s
        # ------------------------------------------------------------
        units = list(acfg.G_ns)
        self.register_buffer("cp0", t([u.cp0 for u in units]))
        self.register_buffer("cp1", t([u.cp1 for u in units]))
        self.register_buffer("cp2", t([u.cp2 for u in units]))
        self.cp_s = (G_s.cp0, G_s.cp1, G_s.cp2)

        def f_g(c, p_g):
            return c[0] + c[1] * p_g + c[2] * p_g ** 2

        f_max = sum(
            f_g((u.cp0, u.cp1, u.cp2), u.P_max) for u in (*units, G_s)
        )
        self.f_max = float(f_max)
        self.w_r = 1.0 / self.f_max

        # ------------------------------------------------------------
        # Constraint names, in the order of z
        # ------------------------------------------------------------
        self.constraint_names = (
            [f"v[bus{b}]" for b in B_q]
            + ["p_g[slack]"]
            + [f"q_g[bus{b}]" for b in B_vs]
            + [f"i_f[line{k}]" for k in limited]
            + [f"i_t[line{k}]" for k in limited]
        )

    # ------------------------------------------------------------------

    def _line_currents(self, v_re, v_im):
        """|Y_f v| I_base and |Y_t v| I_base in kA for all lines."""

        def mag(Y_br_re, Y_br_im, I_base):
            i_re = v_re @ Y_br_re.T - v_im @ Y_br_im.T
            i_im = v_im @ Y_br_re.T + v_re @ Y_br_im.T
            return torch.sqrt(i_re ** 2 + i_im ** 2) * I_base

        return (
            mag(self.Y_f_re, self.Y_f_im, self.I_base_f),
            mag(self.Y_t_re, self.Y_t_im, self.I_base_t),
        )

    @property
    def ramping(self) -> bool:
        return self.dhelm.ramp is not None

    def forward(
        self,
        s: torch.Tensor,
        a_hat: torch.Tensor,
        p_g_prev: torch.Tensor | None = None,
        p_g_s_prev: torch.Tensor | None = None,
        N: int | None = None,
    ) -> CriticOutput:
        """
        Evaluate the critic.

        p_g_prev:
            (B, n_g) previous-step p_g of G_ns, MW (Eq. 3 via Eq. 18).
        p_g_s_prev:
            (B,) previous-step slack output p*_g of (16b), MW.
        Both are only used when the D-HELM has a RampConfig.
        N:
            Passed to the D-HELM (fixed number of terms; for tests).
        """

        cfg = self.config
        base = self.dhelm.base_mva
        B_s = self.dhelm.B_s

        p_g_prev_pu = None
        if self.ramping and p_g_prev is not None:
            p_g_prev_pu = torch.as_tensor(p_g_prev, dtype=DTYPE) / base

        out = self.dhelm(s, a_hat, N=N, p_g_prev=p_g_prev_pu)
        specs = out.specs

        # ---- (15) slack p_g: injection minus everything else on the bus
        p_g_s = base * (out.e_re[:, B_s] - specs.p[:, B_s])

        # ---- (16c) unit q_g per voltage bus
        B_vs = self.B_vs_idx
        q_g = base * (
            out.e_im[:, B_vs] - specs.q[:, B_vs] + self.q_ctrl[B_vs]
        )

        # ---- (16a) PQ voltages
        v_q = out.v_abs[:, self.dhelm.B_q_idx]

        # ---- (16d)-(16e) line currents
        i_f, i_t = self._line_currents(out.v_re, out.v_im)

        # ---- (5) cost and reward
        p_g = base * specs.p_g
        c0, c1, c2 = self.cp_s
        f_sum = (
            (self.cp0 + self.cp1 * p_g + self.cp2 * p_g ** 2).sum(dim=1)
            + c0 + c1 * p_g_s + c2 * p_g_s ** 2
        )
        r = cfg.w_0 - self.w_r * f_sum

        # ---- (6) violations
        P_lo = torch.full_like(p_g_s, self.P_min_s)
        P_hi = torch.full_like(p_g_s, self.P_max_s)

        if self.ramping and p_g_s_prev is not None:
            prev = torch.as_tensor(p_g_s_prev, dtype=DTYPE,
                                   device=p_g_s.device).reshape(-1)
            P_lo = torch.maximum(P_lo, prev - self.dhelm.R_d_s)
            P_hi = torch.minimum(P_hi, prev + self.dhelm.R_u_s)

        eta_p = 1.0 / (self.P_max_s - self.P_min_s)

        i_max = self.i_max
        li = self.line_idx

        z = torch.cat(
            [
                violation(v_q, self.v_min, self.v_max,
                          1.0 / (self.v_max - self.v_min)),
                violation(p_g_s, P_lo, P_hi, eta_p).unsqueeze(1),
                violation(q_g, self.q_g_min, self.q_g_max,
                          1.0 / (self.q_g_max - self.q_g_min)),
                violation(i_f[:, li], -i_max, i_max, 0.5 / i_max),
                violation(i_t[:, li], -i_max, i_max, 0.5 / i_max),
            ],
            dim=1,
        )

        return CriticOutput(
            helm=out,
            p_g=p_g,
            p_g_s=p_g_s,
            q_g=q_g,
            v_q=v_q,
            i_f=i_f,
            i_t=i_t,
            f_sum=f_sum,
            r=r,
            z=z,
        )

    def describe(self) -> str:
        """Constraint list with bounds and eta."""

        lines = [
            "D-HELM critic constraints (Eq. 6)",
            f"  reward (Eq. 5): r = {self.config.w_0} - sum f_g / {self.f_max:.1f}",
        ]
        bounds = (
            [(lo, hi) for lo, hi in zip(self.v_min.tolist(), self.v_max.tolist())]
            + [(self.P_min_s, self.P_max_s)]
            + [(lo, hi) for lo, hi in zip(self.q_g_min.tolist(), self.q_g_max.tolist())]
            + [(-i, i) for i in self.i_max.tolist()] * 2
        )
        for name, (lo, hi) in zip(self.constraint_names, bounds):
            lines.append(
                f"  {name:16s} [{lo:9.4f}, {hi:9.4f}]  eta = {1 / (hi - lo):.4g}"
            )
        return "\n".join(lines)
