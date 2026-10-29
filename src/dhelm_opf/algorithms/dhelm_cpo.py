"""
Constrained policy optimization with the D-HELM critic.

Paper basis
-----------
Eq. (19), Lagrangian relaxation of (7):

    min_{lam >= 0} max_pi  E[ sum_t r_t  -  lam^T sum_t z_t ]

so the policy minimizes, per minibatch (Appendix A.1),

    loss = - mean(r)  +  sum_n lam_n * mean(z_n)

Eq. (20), multiplier update after each policy update:

    lam_n^(k+1) = Gamma_lam[ lam_n^(k) + eta_lam * E[z_n] ],
    Gamma_lam projects onto [0, lam_max]

Table 1: lam^(0) = 10, Adam, lr 1e-3, minibatch 32 (5-bus).

Because the critic is exact and differentiable, the policy gradient is
obtained by back-propagating the loss through the critic (D-HELM) into
the policy; there is no critic training step and no target network.

Notation follows the paper (see NOTATION.md).

Reproduction choices
--------------------
* eta_lam = 1.0 and lam_max = 1000 (not given in the paper).
* As written, Eq. (20) never decreases lam (z >= 0).  This is kept
  as specified; lam is logged so its growth can be monitored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dhelm_opf.models.dhelm_critic import DHELMCritic
from dhelm_opf.models.policy import DHelmPolicy


@dataclass(frozen=True)
class CPOConfig:
    lr: float = 1e-3
    batch_size: int = 32
    window: int = 1
    lam_0: float = 10.0
    eta_lam: float = 1.0
    lam_max: float = 1000.0
    grad_clip: float | None = None


@dataclass
class Evaluation:
    """Critic results for a batch of windows, flattened over time."""

    a_hat: torch.Tensor       # (B, L, action_dim)  normalized actions
    r: torch.Tensor           # (B * L,)  Eq. (5)
    f_sum: torch.Tensor       # (B * L,)  sum_g f_g(p_g), $/h
    z: torch.Tensor           # (B * L, n_constraint)  Eq. (6)
    converged: torch.Tensor   # (B * L,) bool, D-HELM converged
    N: int                    # D-HELM series terms used
    p_g: torch.Tensor | None = None     # (B * L, n_g) MW
    p_g_s: torch.Tensor | None = None   # (B * L,) MW


class DHelmCPO:
    """Policy, multipliers and optimizer, with one-call training steps."""

    def __init__(
        self,
        policy: DHelmPolicy,
        critic: DHELMCritic,
        config: CPOConfig | None = None,
    ) -> None:

        self.policy = policy
        self.critic = critic
        self.config = config or CPOConfig()

        n_con = len(critic.constraint_names)
        self.lam = torch.full(
            (n_con,), self.config.lam_0, dtype=torch.float64
        )

        self.optimizer = torch.optim.Adam(
            policy.parameters(), lr=self.config.lr
        )

    # ------------------------------------------------------------------

    def evaluate(self, s: torch.Tensor) -> Evaluation:
        """
        Run the policy over (B, L, d) windows of states s and evaluate
        every step with the critic.  Gradients flow back into the policy.
        """

        if s.dim() == 2:
            s = s.unsqueeze(1)

        if self.critic.ramping:
            return self._rollout_with_ramping(s)

        a_hat = self.policy(s)                        # (B, L, a)
        B, L, _ = a_hat.shape

        # Steps are independent for the critic: evaluate in one batch.
        out = self.critic(
            s.reshape(B * L, -1).to(torch.float64),
            a_hat.reshape(B * L, -1).to(torch.float64),
        )
        return Evaluation(
            a_hat=a_hat,
            r=out.r,
            f_sum=out.f_sum,
            z=out.z,
            converged=out.helm.converged,
            N=out.helm.N,
            p_g=out.p_g,
            p_g_s=out.p_g_s,
        )

    def _rollout_with_ramping(self, s: torch.Tensor) -> Evaluation:
        """
        Step-by-step rollout over each window (Sec. 3.1, Eq. 3):

            a_hat_t = pi(s_t, p_{g,t-1}),
            p_{g,t} in [max(P_min, p_{g,t-1} - R_d), min(P_max, p_{g,t-1} + R_u)]
            slack (16b) uses p*_g = p_{g_s,t-1}

        The policy's extra input is the previous p_g of G_ns scaled by
        the static range [P_min, P_max] to [-1, 1] (0 at t = 0, when
        there is no previous step and no ramp limit).  Gradients flow
        through p_{g,t-1}, so z_t depends on a_t and a_{t-1} as in the
        paper.
        """

        policy = self.policy
        acfg = self.critic.dhelm.action_config
        B, L, _ = s.shape

        s = s.to(policy.state_low.dtype)
        P_min = torch.as_tensor(acfg.P_min, dtype=torch.float64)
        P_max = torch.as_tensor(acfg.P_max, dtype=torch.float64)

        hc = None
        prev_in = s.new_zeros(B, policy.n_g)
        p_g_prev, p_g_s_prev = None, None

        a_hat, r, f_sum, z, conv, p_g, p_g_s = [], [], [], [], [], [], []
        N = 0

        for t in range(L):
            a_hat_t, hc = policy.step(s[:, t], prev_in, hc)
            out = self.critic(
                s[:, t].to(torch.float64),
                a_hat_t.to(torch.float64),
                p_g_prev=p_g_prev,
                p_g_s_prev=p_g_s_prev,
            )
            a_hat.append(a_hat_t)
            r.append(out.r)
            f_sum.append(out.f_sum)
            z.append(out.z)
            conv.append(out.helm.converged)
            p_g.append(out.p_g)
            p_g_s.append(out.p_g_s)
            N = max(N, out.helm.N)

            p_g_prev, p_g_s_prev = out.p_g, out.p_g_s
            prev_in = (2.0 * (out.p_g - P_min) / (P_max - P_min) - 1.0).to(s.dtype)

        # Order (b, t), matching the reshape of the non-ramping path.
        return Evaluation(
            a_hat=torch.stack(a_hat, dim=1),
            r=torch.stack(r, dim=1).reshape(-1),
            f_sum=torch.stack(f_sum, dim=1).reshape(-1),
            z=torch.stack(z, dim=1).reshape(B * L, -1),
            converged=torch.stack(conv, dim=1).reshape(-1),
            N=N,
            p_g=torch.stack(p_g, dim=1).reshape(B * L, -1),
            p_g_s=torch.stack(p_g_s, dim=1).reshape(-1),
        )

    def loss(self, ev: Evaluation) -> torch.Tensor:
        """Eq. (19): -mean(r) + sum_n lam_n mean(z_n)."""

        return -ev.r.mean() + (self.lam * ev.z.mean(dim=0)).sum()

    def update_lam(self, z_mean: torch.Tensor) -> None:
        """Eq. (20): lam <- Gamma_lam[lam + eta_lam E[z]]."""

        cfg = self.config
        self.lam = torch.clamp(
            self.lam + cfg.eta_lam * z_mean.detach(),
            0.0,
            cfg.lam_max,
        )

    def train_step(self, s: np.ndarray | torch.Tensor) -> dict:
        """
        One policy update on a minibatch of state windows, followed by
        the multiplier update.  Returns scalar metrics for logging.
        """

        s = torch.as_tensor(s, dtype=torch.float32)

        self.policy.train()
        ev = self.evaluate(s)
        loss = self.loss(ev)

        self.optimizer.zero_grad()
        loss.backward()

        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.grad_clip
            )

        self.optimizer.step()

        z_mean = ev.z.mean(dim=0).detach()
        self.update_lam(z_mean)

        return metrics(ev, loss=loss.item(), lam=self.lam,
                       names=self.critic.constraint_names)

    @torch.no_grad()
    def evaluate_metrics(self, s) -> dict:
        """Metrics on a batch without updating anything."""

        s = torch.as_tensor(s, dtype=torch.float32)
        self.policy.eval()
        ev = self.evaluate(s)
        return metrics(ev, loss=self.loss(ev).item(), lam=self.lam,
                       names=self.critic.constraint_names)


def metrics(ev: Evaluation, loss: float, lam, names) -> dict:
    """Scalar summary of an Evaluation (batch means)."""

    z = ev.z.detach()
    z_sum = z.sum(dim=1)

    m = {
        "loss": loss,
        "r": ev.r.detach().mean().item(),
        "f_sum": ev.f_sum.detach().mean().item(),
        "z_sum": z_sum.mean().item(),
        "feasible": (z_sum <= 1e-6).double().mean().item(),
        "helm_converged": ev.converged.double().mean().item(),
        "N": ev.N,
    }
    for k, name in enumerate(names):
        m[f"z/{name}"] = z[:, k].mean().item()
        m[f"lam/{name}"] = float(lam[k])
    return m
