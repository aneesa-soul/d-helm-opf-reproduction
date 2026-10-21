"""
Differentiable, batched D-HELM in PyTorch.

This is the differentiable counterpart of the NumPy reference solver in
recursion.py / solver.py.  It evaluates the same embedding (Eqs. 11-13
of the paper) for a whole minibatch of (state, action) pairs, so that
the policy gradient can flow from the bus voltages back to the action.

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
from dhelm_opf.helm.recursion import build_coefficient_matrix
from dhelm_opf.helm.solver import HelmSolverConfig
from dhelm_opf.models.action_config import ActionConfig


DTYPE = torch.float64


@dataclass
class DHelmSpecs:
    """
    Per-sample specified quantities (Eq. 17, step 1), all in p.u.

    p_spec, q_spec : (B, n_bus)  net specified injections
                                 (q_spec is only used at PQ buses)
    v_sp           : (B, n_bus)  voltage-magnitude setpoints
                                 (only PV and slack entries are used)
    """

    p_spec: torch.Tensor
    q_spec: torch.Tensor
    v_sp: torch.Tensor


@dataclass
class DHelmOutput:
    """
    Result of one batched D-HELM evaluation.

    All tensors carry gradients with respect to the inputs except
    `mismatch`, `converged` and `n_terms`.
    """

    v_re: torch.Tensor          # (B, n_bus)
    v_im: torch.Tensor          # (B, n_bus)
    v_mag: torch.Tensor         # (B, n_bus)

    e_re: torch.Tensor          # (B, n_bus)  P injection, p.u.
    e_im: torch.Tensor          # (B, n_bus)  Q injection, p.u.

    q_pv: torch.Tensor          # (B, n_pv)   sum_n Q_n at PV buses

    specs: DHelmSpecs

    n_terms: int
    mismatch: torch.Tensor      # (B,)  Eq. (14) mismatch, no grad
    converged: torch.Tensor     # (B,)  bool


class DHelmNetwork(torch.nn.Module):
    """
    Constant network data and the batched D-HELM recursion.

    Everything that does not depend on the action or the state is built
    once here: Ybus, the Y^tr / Y^sh split, A and its LU factors, bus
    index sets, and the incidence matrices that map generators, wind and
    loads onto buses.

    Parameters
    ----------
    net:
        Base pandapower network (without wind units; wind comes from the
        state).  It is copied, not modified.
    action_config:
        Action layout derived from the same network.
    uncertainty_config:
        Wind buses / ratings and the state layout.
    solver_config:
        n_max, tolerance and min_order.  Defaults follow Algorithm 1.
    """

    def __init__(
        self,
        net: pp.pandapowerNet,
        action_config: ActionConfig,
        uncertainty_config: UncertaintyConfig | None = None,
        solver_config: HelmSolverConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:

        super().__init__()

        self.action_config = action_config
        self.uncertainty_config = uncertainty_config or UncertaintyConfig()
        self.solver_config = solver_config or HelmSolverConfig()

        net = copy.deepcopy(net)

        # ------------------------------------------------------------
        # Ybus / bus types from the same path the NumPy solver uses.
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
        A, _, y_sh = build_coefficient_matrix(problem)

        n_bus = problem.ybus.shape[0]

        self.n_bus = n_bus
        self.slack = problem.slack_bus
        self.pv = list(problem.pv_buses)
        self.pq = list(problem.pq_buses)
        self.n_pv = len(self.pv)

        expected_pv = sorted(
            b for b in action_config.voltage_buses
            if b != action_config.slack.bus
        )
        if sorted(self.pv) != expected_pv or self.slack != action_config.slack.bus:
            raise ValueError(
                "ActionConfig voltage buses do not match the network's "
                "PV / slack buses."
            )

        def t(x):
            return torch.as_tensor(np.asarray(x), dtype=DTYPE, device=device)

        LU, pivots = torch.linalg.lu_factor(t(A))
        self.register_buffer("LU", LU)
        self.register_buffer("pivots", pivots)

        self.register_buffer("ysh_g", t(np.real(y_sh)))
        self.register_buffer("ysh_b", t(np.imag(y_sh)))
        self.register_buffer("Y_g", t(np.real(problem.ybus)))
        self.register_buffer("Y_b", t(np.imag(problem.ybus)))

        base = float(net.sn_mva)
        self.base_mva = base

        # ------------------------------------------------------------
        # Action scaling (same linear map as ActionDecoder)
        # ------------------------------------------------------------
        cfg = action_config

        self.register_buffer("v_lo", t(cfg.voltage_min))
        self.register_buffer("v_hi", t(cfg.voltage_max))
        self.register_buffer("p_lo", t(cfg.generator_p_min_mw / base))
        self.register_buffer("p_hi", t(cfg.generator_p_max_mw / base))

        v_map = np.zeros((cfg.n_voltage, n_bus))
        for k, bus in enumerate(cfg.voltage_buses):
            v_map[k, bus] = 1.0
        self.register_buffer("V_map", t(v_map))

        # Generator -> bus, including sgen `scaling` (as in build_sbus).
        c_gen = np.zeros((cfg.n_generator, n_bus))
        for k, unit in enumerate(cfg.generators):
            scaling = (
                float(net.sgen.at[unit.index, "scaling"])
                if unit.table == "sgen" else 1.0
            )
            c_gen[k, unit.bus] = scaling
        self.register_buffer("C_gen", t(c_gen))

        # ------------------------------------------------------------
        # Fixed injections: every in-service sgen that is neither a
        # controllable unit nor wind (P and Q), plus the fixed Q of the
        # controllable sgens.  Gen Q is never fixed (solved by HELM).
        # ------------------------------------------------------------
        controlled_sgens = {
            u.index for u in cfg.generators if u.table == "sgen"
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
        # Loads and wind (the state vector)
        #
        # state = [p_d scale (n_load), q_d scale (n_load), wind scale (n_w)]
        # ------------------------------------------------------------
        loads = net.load[net.load["in_service"]]
        n_load = len(loads)

        c_load = np.zeros((n_load, n_bus))
        for k, bus in enumerate(loads["bus"].astype(int)):
            c_load[k, bus] = 1.0

        self.register_buffer("C_load", t(c_load))
        self.register_buffer("p_load_nom", t(loads["p_mw"].to_numpy() / base))
        self.register_buffer("q_load_nom", t(loads["q_mvar"].to_numpy() / base))

        wind_buses = self.uncertainty_config.wind_buses
        wind_rated = self.uncertainty_config.wind_rated_mw

        c_wind = np.zeros((len(wind_buses), n_bus))
        for k, bus in enumerate(wind_buses):
            c_wind[k, bus] = 1.0

        self.register_buffer("C_wind", t(c_wind))
        self.register_buffer("wind_rated", t(np.asarray(wind_rated) / base))

        self.n_load = n_load
        self.n_wind = len(wind_buses)
        self.state_dim = 2 * n_load + self.n_wind
        self.action_dim = cfg.action_dim

        pq = torch.as_tensor(self.pq, dtype=torch.long, device=device)
        pv = torch.as_tensor(self.pv, dtype=torch.long, device=device)
        self.register_buffer("pq_idx", pq)
        self.register_buffer("pv_idx", pv)

    # ------------------------------------------------------------------
    # Eq. (17), step 1: linear assignment
    # ------------------------------------------------------------------

    def specs(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> DHelmSpecs:
        """
        Map (state, normalized action) to specified injections and
        voltage setpoints, in p.u.
        """

        state = state.to(DTYPE)
        action = action.to(DTYPE)

        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state dimension {self.state_dim}, "
                f"got {state.shape[-1]}."
            )
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action dimension {self.action_dim}, "
                f"got {action.shape[-1]}."
            )

        n_load, n_wind = self.n_load, self.n_wind
        n_v = self.action_config.n_voltage

        p_scale = state[:, :n_load]
        q_scale = state[:, n_load:2 * n_load]
        w_scale = state[:, 2 * n_load:2 * n_load + n_wind]

        a_v = action[:, :n_v]
        a_p = action[:, n_v:]

        v_set = self.v_lo + 0.5 * (a_v + 1.0) * (self.v_hi - self.v_lo)
        p_gen = self.p_lo + 0.5 * (a_p + 1.0) * (self.p_hi - self.p_lo)

        p_spec = (
            p_gen @ self.C_gen
            + (w_scale * self.wind_rated) @ self.C_wind
            - (p_scale * self.p_load_nom) @ self.C_load
            + self.p_fix
        )

        q_spec = (
            - (q_scale * self.q_load_nom) @ self.C_load
            + self.q_fix
        )

        v_sp = v_set @ self.V_map

        return DHelmSpecs(p_spec=p_spec, q_spec=q_spec, v_sp=v_sp)

    # ------------------------------------------------------------------
    # Helpers in real arithmetic
    # ------------------------------------------------------------------

    def injections(self, v_re, v_im):
        """e = v * conj(Y v) with the full Ybus (real / imag parts)."""

        i_re = v_re @ self.Y_g.T - v_im @ self.Y_b.T
        i_im = v_im @ self.Y_g.T + v_re @ self.Y_b.T

        e_re = v_re * i_re + v_im * i_im
        e_im = v_im * i_re - v_re * i_im

        return e_re, e_im

    def eq14_mismatch(self, v_re, v_im, specs: DHelmSpecs) -> torch.Tensor:
        """Per-sample max of the Eq. (14a)-(14c) mismatches."""

        e_re, e_im = self.injections(v_re, v_im)

        pq, pv = self.pq_idx, self.pv_idx

        pq_err = torch.sqrt(
            (e_re[:, pq] - specs.p_spec[:, pq]) ** 2
            + (e_im[:, pq] - specs.q_spec[:, pq]) ** 2
        )
        v_mag = torch.sqrt(v_re[:, pv] ** 2 + v_im[:, pv] ** 2)
        pv_v_err = torch.abs(v_mag - specs.v_sp[:, pv])
        pv_p_err = torch.abs(e_re[:, pv] - specs.p_spec[:, pv])

        parts = [pv_v_err, pv_p_err]
        if len(self.pq) > 0:
            parts.append(pq_err)

        return torch.cat(parts, dim=1).amax(dim=1)

    # ------------------------------------------------------------------
    # Eqs. (11)-(13): batched recursion
    # ------------------------------------------------------------------

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        n_terms: int | None = None,
    ) -> DHelmOutput:
        """
        Evaluate D-HELM for a batch.

        Parameters
        ----------
        state:  (B, state_dim)  demand / wind scaling factors
        action: (B, action_dim) normalized policy action in [-1, 1]
        n_terms:
            If given, compute exactly this many terms (no early stop).
            Used for exact comparison with the NumPy solver and for
            gradcheck.  If None, stop per Algorithm 1 / Eq. (14).
        """

        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        specs = self.specs(state, action)

        B = state.shape[0]
        n_bus = self.n_bus
        pq, pv, s = self.pq_idx, self.pv_idx, self.slack

        cfg = self.solver_config
        max_order = n_terms if n_terms is not None else cfg.max_order

        P = specs.p_spec
        Q_sp = specs.q_spec

        ones = torch.ones(B, n_bus, dtype=DTYPE, device=P.device)
        zeros = torch.zeros(B, n_bus, dtype=DTYPE, device=P.device)

        # No-load germ: V_0 = 1, W_0 = 1, Q_0 = 0.
        V_re, V_im = [ones], [zeros]
        W_re, W_im = [ones], [zeros]
        Qs = [torch.zeros(B, self.n_pv, dtype=DTYPE, device=P.device)]

        v_re_sum, v_im_sum = ones, zeros

        mismatch = None
        converged = torch.zeros(B, dtype=torch.bool, device=P.device)
        orders_used = 0

        for n in range(1, max_order + 1):

            Wr, Wi = W_re[n - 1], W_im[n - 1]
            Vr, Vi = V_re[n - 1], V_im[n - 1]

            # Y^sh V_{n-1}
            ysh_re = self.ysh_g * Vr - self.ysh_b * Vi
            ysh_im = self.ysh_g * Vi + self.ysh_b * Vr

            # ---- R_n, PQ buses: conj(S) conj(W_{n-1}) ----------------
            r_re_pq = P[:, pq] * Wr[:, pq] - Q_sp[:, pq] * Wi[:, pq]
            r_im_pq = -P[:, pq] * Wi[:, pq] - Q_sp[:, pq] * Wr[:, pq]

            # ---- R_n, PV buses:
            #      P conj(W_{n-1}) - j sum_m Q_m conj(W_{n-m}) ----------
            r_re_pv = P[:, pv] * Wr[:, pv]
            r_im_pv = -P[:, pv] * Wi[:, pv]

            for m in range(1, n):
                Qm = Qs[m]
                r_re_pv = r_re_pv - Qm * W_im[n - m][:, pv]
                r_im_pv = r_im_pv - Qm * W_re[n - m][:, pv]

            # Assemble R_n - Y^sh V_{n-1}, slack row separately.
            b_re = torch.zeros(B, n_bus, dtype=DTYPE, device=P.device)
            b_im = torch.zeros(B, n_bus, dtype=DTYPE, device=P.device)

            b_re = b_re.index_copy(1, pq, r_re_pq - ysh_re[:, pq])
            b_im = b_im.index_copy(1, pq, r_im_pq - ysh_im[:, pq])
            b_re = b_re.index_copy(1, pv, r_re_pv - ysh_re[:, pv])
            b_im = b_im.index_copy(1, pv, r_im_pv - ysh_im[:, pv])

            if n == 1:
                slack_val = (specs.v_sp[:, s] - 1.0).unsqueeze(1)
                b_re = torch.cat(
                    [b_re[:, :s], slack_val, b_re[:, s + 1:]], dim=1
                )

            # ---- U_n, PV voltage equations --------------------------
            if n == 1:
                U = 0.5 * (specs.v_sp[:, pv] ** 2 - 1.0)
            else:
                acc = torch.zeros(B, self.n_pv, dtype=DTYPE, device=P.device)
                for m in range(1, n):
                    acc = acc + (
                        V_re[m][:, pv] * V_re[n - m][:, pv]
                        + V_im[m][:, pv] * V_im[n - m][:, pv]
                    )
                U = -0.5 * acc

            b = torch.cat([b_re, b_im, U], dim=1)            # (B, 2n+m)

            # ---- x_n = A^{-1} b_n (one LU solve for the batch) -------
            x = torch.linalg.lu_solve(self.LU, self.pivots, b.T).T

            Vr_n = x[:, :n_bus]
            Vi_n = x[:, n_bus:2 * n_bus]
            Q_n = x[:, 2 * n_bus:]

            V_re.append(Vr_n)
            V_im.append(Vi_n)
            Qs.append(Q_n)

            # ---- Eq. (11): W_n = -sum_{m=1}^{n} V_m W_{n-m}  (V_0 = 1)
            w_re = torch.zeros_like(Vr_n)
            w_im = torch.zeros_like(Vr_n)
            for m in range(1, n + 1):
                w_re = w_re - (V_re[m] * W_re[n - m] - V_im[m] * W_im[n - m])
                w_im = w_im - (V_re[m] * W_im[n - m] + V_im[m] * W_re[n - m])
            W_re.append(w_re)
            W_im.append(w_im)

            v_re_sum = v_re_sum + Vr_n
            v_im_sum = v_im_sum + Vi_n
            orders_used = n

            # ---- Algorithm 1, step 4: Eq. (14) check, outside graph --
            with torch.no_grad():
                mismatch = self.eq14_mismatch(
                    v_re_sum.detach(), v_im_sum.detach(), specs
                )
                converged = mismatch <= cfg.tolerance

            if (
                n_terms is None
                and n >= cfg.min_order
                and bool(converged.all())
            ):
                break

        e_re, e_im = self.injections(v_re_sum, v_im_sum)
        v_mag = torch.sqrt(v_re_sum ** 2 + v_im_sum ** 2)
        q_pv = torch.stack(Qs, dim=0).sum(dim=0)

        return DHelmOutput(
            v_re=v_re_sum,
            v_im=v_im_sum,
            v_mag=v_mag,
            e_re=e_re,
            e_im=e_im,
            q_pv=q_pv,
            specs=specs,
            n_terms=orders_used,
            mismatch=mismatch,
            converged=converged,
        )
