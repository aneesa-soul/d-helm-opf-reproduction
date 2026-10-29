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

        a_hat = self.policy(s)                        # (B, L, a)
        B, L, _ = a_hat.shape

        ccfg = self.critic.config
        ramping = ccfg.R_u is not None or ccfg.R_d is not None

        if not ramping:
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
            )

        # With ramping, (16b) at step t needs the slack output of t-1.
        r, f_sum, z, conv = [], [], [], []
        p_g_s_prev = None
        N = 0

        for t in range(L):
            out = self.critic(
                s[:, t].to(torch.float64),
                a_hat[:, t].to(torch.float64),
                p_g_s_prev=p_g_s_prev,
            )
            r.append(out.r)
            f_sum.append(out.f_sum)
            z.append(out.z)
            conv.append(out.helm.converged)
            p_g_s_prev = out.p_g_s
            N = max(N, out.helm.N)

        # Order (b, t) to match the reshape used above.
        return Evaluation(
            a_hat=a_hat,
            r=torch.stack(r, dim=1).reshape(-1),
            f_sum=torch.stack(f_sum, dim=1).reshape(-1),
            z=torch.stack(z, dim=1).reshape(B * L, -1),
            converged=torch.stack(conv, dim=1).reshape(-1),
            N=N,
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
