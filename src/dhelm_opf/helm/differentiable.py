"""
Differentiable, batched D-HELM in PyTorch.

This is the differentiable counterpart of the NumPy reference solver in
recursion.py / solver.py.  It evaluates the same embedding (Eqs. 9-13
of the paper) for a whole minibatch of (s, a_hat) pairs, so that the
policy gradient can flow from the bus voltages back to the action.

Paper basis
-----------
Sec. 3.2.2 and Eq. (17):

    1: (|v|^sp, e^sp, P) <- (a, s),   V_0 <- 1      linear assignment
    2: repeat n = 1..N:
           b_n = f(V_0..V_{n-1}, a, s)              Eqs. (12)-(13)
           x_n = A^{-1} b_n                         Eq. (12)
    3: v = sum_n V_n,  e = v * conj(Y v)            linear functions

with A decomposed once (LU) for all coefficients and samples, and N the
number of terms after Algorithm 1 has converged (Eq. 14).

Notation follows the paper (see NOTATION.md).  Complex quantities are
carried as real and imaginary parts (suffixes _re, _im).

Implementation choices
----------------------
* Real arithmetic throughout (the Re/Im split of Eq. 12), so no complex
  autograd is involved.
* float64 inside the solver; the Eq. (14) tolerance and the validation
  tests need it.  Inputs of other dtypes are cast.
* Stopping: after every order the Eq. (14) mismatches are checked under
  torch.no_grad(); the loop stops once EVERY sample in the batch meets
  the tolerance (or at n_max).  The stopping decision is outside the
  graph, so the gradient is exactly that of the N-term series.
  Samples that converge earlier simply receive a few extra terms.
* No PV->PQ switching at reactive limits.  As in the paper, generator
  Q limits are handled as constraint violations (Eq. 16c) by the
  critic, not inside the power flow.

Constraint violations, slack P/Q, line currents and the reward
(Eqs. 15, 16, 6, 5) are NOT computed here; that is the critic's job.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandapower as pp
import torch

from dhelm_opf.environment.uncertainty import (
    UncertaintyConfig,
    WIND_SGEN_TYPE,
)
from dhelm_opf.helm.problem import build_helm_problem_from_network
from dhelm_opf.helm.recursion import build_A
from dhelm_opf.helm.solver import HelmSolverConfig
from dhelm_opf.models.action_config import ActionConfig, RampConfig


DTYPE = torch.float64


@dataclass
class DHelmSpecs:
    """
    Per-sample specified quantities (Eq. 17, step 1), all in p.u.

    p, q  : (B, n_bus)  specified net injection e^sp = p + j q (Eq. 8);
                        q is only used at PQ buses
    v_sp  : (B, n_bus)  |v_i|^sp (only B_v and B_s entries are used)
    p_g   : (B, n_g)    p_g of the non-slack units G_ns, in ActionConfig
                        order
    """

    p: torch.Tensor
    q: torch.Tensor
    v_sp: torch.Tensor
    p_g: torch.Tensor


@dataclass
class DHelmOutput:
    """
    Result of one batched D-HELM evaluation.

    All tensors carry gradients with respect to the inputs except
    `mismatch`, `converged` and `N`.
    """

    v_re: torch.Tensor          # (B, n_bus)  Re(v_i),  v = sum_n V_n
    v_im: torch.Tensor          # (B, n_bus)  Im(v_i)
    v_abs: torch.Tensor         # (B, n_bus)  |v_i|

    e_re: torch.Tensor          # (B, n_bus)  Re(e_i) = p_i, p.u.
    e_im: torch.Tensor          # (B, n_bus)  Im(e_i) = q_i, p.u.

    q_v: torch.Tensor           # (B, |B_v|)  sum_n Q_{i,n}, i in B_v

    specs: DHelmSpecs

    N: int                      # series terms used
    mismatch: torch.Tensor      # (B,)  Eq. (14) mismatch, no grad
    converged: torch.Tensor     # (B,)  bool


class DHelmNetwork(torch.nn.Module):
    """
    Constant network data and the batched D-HELM recursion.

    Everything that does not depend on the action or the state is built
    once here: Y, the Y_tr / Y_sh split, A and its LU factors, bus
    index sets, and the incidence matrices that map units, wind and
    loads onto buses.

    Parameters
    ----------
    net:
        Base pandapower network (without wind units; wind comes from the
        state).  It is copied, not modified.
    action_config:
        Action layout derived from the same network.
    uncertainty_config:
        Wind buses B_w / ratings and the state layout.
    solver_config:
        n_max, epsilon and n_min.  Defaults follow Algorithm 1.
    ramp:
        Ramp limits (Eq. 3).  When given, the p_g actions of step t are
        scaled into the window [max(P_min, p_{g,t-1} - R_d),
        min(P_max, p_{g,t-1} + R_u)] (Eq. 18 with time-varying a_min,
        a_max) whenever the previous output p_g_prev is passed.
    """

    def __init__(
        self,
        net: pp.pandapowerNet,
        action_config: ActionConfig,
        uncertainty_config: UncertaintyConfig | None = None,
        solver_config: HelmSolverConfig | None = None,
        device: torch.device | str = "cpu",
        ramp: RampConfig | None = None,
    ) -> None:

        super().__init__()

        self.action_config = action_config
        self.uncertainty_config = uncertainty_config or UncertaintyConfig()
        self.solver_config = solver_config or HelmSolverConfig()
        self.ramp = ramp

        net = copy.deepcopy(net)

        # ------------------------------------------------------------
        # Y / bus types from the same path the NumPy solver uses.
        # ------------------------------------------------------------
        pp.runpp(net)

        lookup = net._pd2ppc_lookups["bus"]
        if not np.array_equal(lookup[net.bus.index], np.arange(len(net.bus))):
            raise ValueError(
                "pandapower internal bus order differs from net.bus order; "
                "a bus permutation would be required."
            )

        if float(net.ext_grid.iloc[0]["va_degree"]) != 0.0:
            raise ValueError(
                "The paper's slack row (|v_sp| - 1) assumes a 0° slack angle."
            )

        problem = build_helm_problem_from_network(net)
        A, _, Y_sh = build_A(problem)

        n_bus = problem.Y.shape[0]

        self.n_bus = n_bus
        self.B_s = problem.B_s
        self.B_v = list(problem.B_v)
        self.B_q = list(problem.B_q)

        cfg = action_config

        expected_B_v = sorted(b for b in cfg.B_vs if b != cfg.G_s.bus)
        if sorted(self.B_v) != expected_B_v or self.B_s != cfg.G_s.bus:
            raise ValueError(
                "ActionConfig voltage buses do not match the network's "
                "PV / slack buses."
            )

        def t(x):
            # np.array(...) copies: pandas can hand out read-only arrays,
            # which torch.as_tensor would otherwise share (UserWarning).
            return torch.tensor(np.array(x, dtype=float), dtype=DTYPE, device=device)

        # A is decomposed once for all coefficients and samples (Eq. 12).
        A_LU, A_piv = torch.linalg.lu_factor(t(A))
        self.register_buffer("A_LU", A_LU)
        self.register_buffer("A_piv", A_piv)

        self.register_buffer("Y_sh_re", t(np.real(Y_sh)))
        self.register_buffer("Y_sh_im", t(np.imag(Y_sh)))
        self.register_buffer("Y_re", t(np.real(problem.Y)))
        self.register_buffer("Y_im", t(np.imag(problem.Y)))

        base = float(net.sn_mva)
        self.base_mva = base

        # ------------------------------------------------------------
        # Eq. (18) bounds a_min, a_max in action order
        # (|v| in p.u., p_g in p.u.), same map as ActionDecoder.
        # ------------------------------------------------------------
        self.n_v = cfg.n_v
        self.n_g = cfg.n_g

        a_min = np.zeros(cfg.action_dim)
        a_max = np.zeros(cfg.action_dim)
        a_min[list(cfg.voltage_action_indices)] = cfg.v_min
        a_max[list(cfg.voltage_action_indices)] = cfg.v_max
        a_min[list(cfg.generator_action_indices)] = cfg.P_min / base
        a_max[list(cfg.generator_action_indices)] = cfg.P_max / base

        self.register_buffer("a_min", t(a_min))
        self.register_buffer("a_max", t(a_max))

        self.register_buffer(
            "idx_v", torch.as_tensor(cfg.voltage_action_indices,
                                     dtype=torch.long, device=device)
        )
        self.register_buffer(
            "idx_p", torch.as_tensor(cfg.generator_action_indices,
                                     dtype=torch.long, device=device)
        )

        # Static P limits of G_ns and ramp limits (Eq. 3), p.u.
        self.register_buffer("P_min_g", t(cfg.P_min / base))
        self.register_buffer("P_max_g", t(cfg.P_max / base))

        if ramp is not None:
            self.register_buffer(
                "R_u_g", t([ramp.R_u(u) / base for u in cfg.G_ns])
            )
            self.register_buffer(
                "R_d_g", t([ramp.R_d(u) / base for u in cfg.G_ns])
            )
            # Slack ramp limits in MW, used by the critic (Eq. 16b).
            self.R_u_s = ramp.R_u(cfg.G_s)
            self.R_d_s = ramp.R_d(cfg.G_s)

        # Voltage actions -> buses.
        C_v = np.zeros((cfg.n_v, n_bus))
        for k, bus in enumerate(cfg.B_vs):
            C_v[k, bus] = 1.0
        self.register_buffer("C_v", t(C_v))

        # Units G_ns -> buses (G_i of Eq. 8), including sgen `scaling`
        # (as in build_e_sp).
        C_g = np.zeros((cfg.n_g, n_bus))
        for k, unit in enumerate(cfg.G_ns):
            scaling = (
                float(net.sgen.at[unit.index, "scaling"])
                if unit.table == "sgen" else 1.0
            )
            C_g[k, unit.bus] = scaling
        self.register_buffer("C_g", t(C_g))

        # ------------------------------------------------------------
        # Fixed injections: every in-service sgen that is neither a
        # controllable unit nor wind (P and Q), plus the fixed Q of the
        # controllable sgens.  Gen Q is never fixed (solved by HELM).
        # ------------------------------------------------------------
        controlled_sgens = {
            u.index for u in cfg.G_ns if u.table == "sgen"
        }

        p_fix = np.zeros(n_bus)
        q_fix = np.zeros(n_bus)

        for index, row in net.sgen.iterrows():
            if not bool(row["in_service"]):
                continue
            if str(row["type"]) == WIND_SGEN_TYPE:
                raise ValueError(
                    "Pass the base network without wind sgens; wind "
                    "comes from the state vector."
                )
            bus = int(row["bus"])
            scaling = float(row["scaling"])
            q_fix[bus] += float(row["q_mvar"]) * scaling / base
            if index not in controlled_sgens:
                p_fix[bus] += float(row["p_mw"]) * scaling / base

        self.register_buffer("p_fix", t(p_fix))
        self.register_buffer("q_fix", t(q_fix))

        # ------------------------------------------------------------
        # Demand and wind (the state s, Eq. 2)
        #
        # s = [p_d scale (n_d), q_d scale (n_d), p_w scale (n_w)]
        # ------------------------------------------------------------
        loads = net.load[net.load["in_service"]]
        n_d = len(loads)

        C_d = np.zeros((n_d, n_bus))
        for k, bus in enumerate(loads["bus"].astype(int)):
            C_d[k, bus] = 1.0

        self.register_buffer("C_d", t(C_d))
        self.register_buffer("p_d_nom", t(loads["p_mw"].to_numpy() / base))
        self.register_buffer("q_d_nom", t(loads["q_mvar"].to_numpy() / base))

        B_w = self.uncertainty_config.B_w
        p_w_rated = self.uncertainty_config.p_w_rated_mw

        C_w = np.zeros((len(B_w), n_bus))
        for k, bus in enumerate(B_w):
            C_w[k, bus] = 1.0

        self.register_buffer("C_w", t(C_w))
        self.register_buffer("p_w_rated", t(np.asarray(p_w_rated) / base))

        self.n_d = n_d
        self.n_w = len(B_w)
        self.state_dim = 2 * n_d + self.n_w
        self.action_dim = cfg.action_dim

        self.register_buffer(
            "B_q_idx", torch.as_tensor(self.B_q, dtype=torch.long, device=device)
        )
        self.register_buffer(
            "B_v_idx", torch.as_tensor(self.B_v, dtype=torch.long, device=device)
        )

    # ------------------------------------------------------------------
    # Eq. (17), step 1: linear assignment
    # ------------------------------------------------------------------

    def specs(
        self,
        s: torch.Tensor,
        a_hat: torch.Tensor,
        p_g_prev: torch.Tensor | None = None,
    ) -> DHelmSpecs:
        """
        Map (s, a_hat) to the specified injections and voltage
        setpoints, in p.u.

        Eq. (18):  a = a_min + (a_max - a_min) (a_hat + 1) / 2
        Eq. (3):   with ramping and p_g_prev (p.u., (B, n_g)) given, the
                   p_g bounds are max(P_min, p_g_prev - R_d) and
                   min(P_max, p_g_prev + R_u)
        Eq. (8):   p_i = sum_{g in G_i} p_g + p_i^w - p_i^d,  q_i = -q_i^d
        """

        s = s.to(DTYPE)
        a_hat = a_hat.to(DTYPE)

        if s.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state dimension {self.state_dim}, "
                f"got {s.shape[-1]}."
            )
        if a_hat.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action dimension {self.action_dim}, "
                f"got {a_hat.shape[-1]}."
            )

        n_d, n_w, n_v = self.n_d, self.n_w, self.n_v

        p_d_scale = s[:, :n_d]
        q_d_scale = s[:, n_d:2 * n_d]
        p_w_scale = s[:, 2 * n_d:2 * n_d + n_w]

        # Eq. (18)
        a = self.a_min + 0.5 * (a_hat + 1.0) * (self.a_max - self.a_min)

        v_set = a[:, self.idx_v]
        p_g = a[:, self.idx_p]

        # Eq. (3): ramp window around the previous output.
        if self.ramp is not None and p_g_prev is not None:
            p_g_prev = p_g_prev.to(DTYPE)
            lo = torch.maximum(self.P_min_g, p_g_prev - self.R_d_g)
            hi = torch.minimum(self.P_max_g, p_g_prev + self.R_u_g)
            p_g = lo + 0.5 * (a_hat[:, self.idx_p] + 1.0) * (hi - lo)

        p = (
            p_g @ self.C_g
            + (p_w_scale * self.p_w_rated) @ self.C_w
            - (p_d_scale * self.p_d_nom) @ self.C_d
            + self.p_fix
        )

        q = (
            - (q_d_scale * self.q_d_nom) @ self.C_d
            + self.q_fix
        )

        v_sp = v_set @ self.C_v

        return DHelmSpecs(
            p=p,
            q=q,
            v_sp=v_sp,
            p_g=p_g,
        )

    # ------------------------------------------------------------------
    # Helpers in real arithmetic
    # ------------------------------------------------------------------

    def compute_e(self, v_re, v_im):
        """e = v * conj(Y v) with the full Y (Algorithm 1, step 3)."""

        i_re = v_re @ self.Y_re.T - v_im @ self.Y_im.T
        i_im = v_im @ self.Y_re.T + v_re @ self.Y_im.T

        e_re = v_re * i_re + v_im * i_im
        e_im = v_im * i_re - v_re * i_im

        return e_re, e_im

    def eq14_mismatch(self, v_re, v_im, specs: DHelmSpecs) -> torch.Tensor:
        """Per-sample max of the Eq. (14a)-(14c) mismatches."""

        e_re, e_im = self.compute_e(v_re, v_im)

        B_q, B_v = self.B_q_idx, self.B_v_idx

        err_14b = torch.sqrt(
            (e_re[:, B_q] - specs.p[:, B_q]) ** 2
            + (e_im[:, B_q] - specs.q[:, B_q]) ** 2
        )
        v_abs_v = torch.sqrt(v_re[:, B_v] ** 2 + v_im[:, B_v] ** 2)
        err_14a = torch.abs(v_abs_v - specs.v_sp[:, B_v])
        err_14c = torch.abs(e_re[:, B_v] - specs.p[:, B_v])

        parts = [err_14a, err_14c]
        if len(self.B_q) > 0:
            parts.append(err_14b)

        return torch.cat(parts, dim=1).amax(dim=1)

    # ------------------------------------------------------------------
    # Eqs. (11)-(13): batched recursion
    # ------------------------------------------------------------------

    def forward(
        self,
        s: torch.Tensor,
        a_hat: torch.Tensor,
        N: int | None = None,
        p_g_prev: torch.Tensor | None = None,
    ) -> DHelmOutput:
        """
        Evaluate D-HELM for a batch.

        Parameters
        ----------
        s:      (B, state_dim)   state (demand / wind scaling factors)
        a_hat:  (B, action_dim)  normalized action tanh(logit) in [-1, 1]
        N:
            If given, compute exactly N terms (no early stop).  Used for
            exact comparison with the NumPy solver and for gradcheck.
            If None, stop per Algorithm 1 / Eq. (14).
        p_g_prev:
            (B, n_g) previous-step p_g of G_ns in p.u. (ramping only).
        """

        if s.dim() == 1:
            s = s.unsqueeze(0)
        if a_hat.dim() == 1:
            a_hat = a_hat.unsqueeze(0)

        specs = self.specs(s, a_hat, p_g_prev)

        batch = s.shape[0]
        n_bus = self.n_bus
        B_q, B_v, B_s = self.B_q_idx, self.B_v_idx, self.B_s
        n_bv = len(self.B_v)

        cfg = self.solver_config
        n_max = N if N is not None else cfg.n_max

        p = specs.p
        q = specs.q
        dev = p.device

        ones = torch.ones(batch, n_bus, dtype=DTYPE, device=dev)
        zeros = torch.zeros(batch, n_bus, dtype=DTYPE, device=dev)

        # No-load germ: V_0 = 1, W_0 = 1, Q_0 = 0.
        V_re, V_im = [ones], [zeros]
        W_re, W_im = [ones], [zeros]
        Q = [torch.zeros(batch, n_bv, dtype=DTYPE, device=dev)]

        v_re, v_im = ones, zeros

        mismatch = None
        converged = torch.zeros(batch, dtype=torch.bool, device=dev)
        N_used = 0

        for n in range(1, n_max + 1):

            W_re_p, W_im_p = W_re[n - 1], W_im[n - 1]
            V_re_p, V_im_p = V_re[n - 1], V_im[n - 1]

            # Y_sh V_{n-1}
            YshV_re = self.Y_sh_re * V_re_p - self.Y_sh_im * V_im_p
            YshV_im = self.Y_sh_re * V_im_p + self.Y_sh_im * V_re_p

            # ---- R_n, PQ buses: conj(e^sp) conj(W_{n-1}) -------------
            R_re_q = p[:, B_q] * W_re_p[:, B_q] - q[:, B_q] * W_im_p[:, B_q]
            R_im_q = -p[:, B_q] * W_im_p[:, B_q] - q[:, B_q] * W_re_p[:, B_q]

            # ---- R_n, PV buses:
            #      P conj(W_{n-1}) - j sum_m Q_m conj(W_{n-m}) ----------
            R_re_v = p[:, B_v] * W_re_p[:, B_v]
            R_im_v = -p[:, B_v] * W_im_p[:, B_v]

            for m in range(1, n):
                Q_m = Q[m]
                R_re_v = R_re_v - Q_m * W_im[n - m][:, B_v]
                R_im_v = R_im_v - Q_m * W_re[n - m][:, B_v]

            # b_n = [Re, Im](R_n - Y_sh V_{n-1}), slack row separately.
            b_re = torch.zeros(batch, n_bus, dtype=DTYPE, device=dev)
            b_im = torch.zeros(batch, n_bus, dtype=DTYPE, device=dev)

            b_re = b_re.index_copy(1, B_q, R_re_q - YshV_re[:, B_q])
            b_im = b_im.index_copy(1, B_q, R_im_q - YshV_im[:, B_q])
            b_re = b_re.index_copy(1, B_v, R_re_v - YshV_re[:, B_v])
            b_im = b_im.index_copy(1, B_v, R_im_v - YshV_im[:, B_v])

            if n == 1:
                R_s = (specs.v_sp[:, B_s] - 1.0).unsqueeze(1)
                b_re = torch.cat(
                    [b_re[:, :B_s], R_s, b_re[:, B_s + 1:]], dim=1
                )

            # ---- U_n, PV voltage equations --------------------------
            if n == 1:
                U_n = 0.5 * (specs.v_sp[:, B_v] ** 2 - 1.0)
            else:
                acc = torch.zeros(batch, n_bv, dtype=DTYPE, device=dev)
                for m in range(1, n):
                    acc = acc + (
                        V_re[m][:, B_v] * V_re[n - m][:, B_v]
                        + V_im[m][:, B_v] * V_im[n - m][:, B_v]
                    )
                U_n = -0.5 * acc

            b_n = torch.cat([b_re, b_im, U_n], dim=1)        # (batch, 2|B|+|B_v|)

            # ---- x_n = A^{-1} b_n (one LU solve for the batch) -------
            x_n = torch.linalg.lu_solve(self.A_LU, self.A_piv, b_n.T).T

            V_re_n = x_n[:, :n_bus]
            V_im_n = x_n[:, n_bus:2 * n_bus]
            Q_n = x_n[:, 2 * n_bus:]

            V_re.append(V_re_n)
            V_im.append(V_im_n)
            Q.append(Q_n)

            # ---- Eq. (11): W_n = -sum_{m=1}^{n} V_m W_{n-m}  (V_0 = 1)
            W_re_n = torch.zeros_like(V_re_n)
            W_im_n = torch.zeros_like(V_re_n)
            for m in range(1, n + 1):
                W_re_n = W_re_n - (V_re[m] * W_re[n - m] - V_im[m] * W_im[n - m])
                W_im_n = W_im_n - (V_re[m] * W_im[n - m] + V_im[m] * W_re[n - m])
            W_re.append(W_re_n)
            W_im.append(W_im_n)

            v_re = v_re + V_re_n
            v_im = v_im + V_im_n
            N_used = n

            # ---- Algorithm 1, step 4: Eq. (14) check, outside graph --
            with torch.no_grad():
                mismatch = self.eq14_mismatch(
                    v_re.detach(), v_im.detach(), specs
                )
                converged = mismatch <= cfg.epsilon

            if (
                N is None
                and n >= cfg.n_min
                and bool(converged.all())
            ):
                break

        e_re, e_im = self.compute_e(v_re, v_im)
        v_abs = torch.sqrt(v_re ** 2 + v_im ** 2)
        q_v = torch.stack(Q, dim=0).sum(dim=0)

        return DHelmOutput(
            v_re=v_re,
            v_im=v_im,
            v_abs=v_abs,
            e_re=e_re,
            e_im=e_im,
            q_v=q_v,
            specs=specs,
            N=N_used,
            mismatch=mismatch,
            converged=converged,
        )
