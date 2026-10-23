"""
Constrained policy optimization with the D-HELM critic.

Paper basis
-----------
Eq. (19), Lagrangian relaxation of (7):

    min_{lambda >= 0} max_pi  E[ sum_t r_t  -  lambda^T sum_t z_t ]

so the policy minimizes, per minibatch (Appendix A.1),

    loss = - mean(r)  +  sum_n lambda_n * mean(z_n)

Eq. (20), multiplier update after each policy update:

    lambda_n <- clip( lambda_n + eta_lambda * E[z_n],  0, lambda_max )

Table 1: lambda^(0) = 10, Adam, lr 1e-3, minibatch 32 (5-bus).

Because the critic is exact and differentiable, the policy gradient is
obtained by back-propagating the loss through the critic (D-HELM) into
the policy; there is no critic training step and no target network.

Reproduction choices
--------------------
* eta_lambda = 1.0 and lambda_max = 1000 (not given in the paper).
* As written, Eq. (20) never decreases lambda (z >= 0).  This is kept
  as specified; lambda is logged so its growth can be monitored.
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
    lambda_init: float = 10.0
    eta_lambda: float = 1.0
    lambda_max: float = 1000.0
    grad_clip: float | None = None


@dataclass
class Evaluation:
    """Critic results for a batch of windows, flattened over time."""

    actions: torch.Tensor     # (B, L, action_dim)
    reward: torch.Tensor      # (B * L,)
    cost: torch.Tensor        # (B * L,)
    z: torch.Tensor           # (B * L, n_constraint)
    converged: torch.Tensor   # (B * L,) bool
    n_terms: int


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
        self.lambdas = torch.full(
            (n_con,), self.config.lambda_init, dtype=torch.float64
        )

        self.optimizer = torch.optim.Adam(
            policy.parameters(), lr=self.config.lr
        )

    # ------------------------------------------------------------------

    def evaluate(self, states: torch.Tensor) -> Evaluation:
        """
        Run the policy over (B, L, d) windows and evaluate every step
        with the critic.  Gradients flow back into the policy.
        """

        if states.dim() == 2:
            states = states.unsqueeze(1)

        actions = self.policy(states)                 # (B, L, a)
        B, L, _ = actions.shape

        ccfg = self.critic.config
        ramping = ccfg.ramp_up_mw is not None or ccfg.ramp_down_mw is not None

        if not ramping:
            # Steps are independent for the critic: evaluate in one batch.
            out = self.critic(
                states.reshape(B * L, -1).to(torch.float64),
                actions.reshape(B * L, -1).to(torch.float64),
            )
            return Evaluation(
                actions=actions,
                reward=out.reward,
                cost=out.cost,
                z=out.z,
                converged=out.helm.converged,
                n_terms=out.helm.n_terms,
            )

        # With ramping, (16b) at step t needs the slack output of t-1.
        rewards, costs, zs, conv = [], [], [], []
        prev_slack = None
        n_terms = 0

        for t in range(L):
            out = self.critic(
                states[:, t].to(torch.float64),
                actions[:, t].to(torch.float64),
                p_slack_prev_mw=prev_slack,
            )
            rewards.append(out.reward)
            costs.append(out.cost)
            zs.append(out.z)
            conv.append(out.helm.converged)
            prev_slack = out.p_slack_mw
            n_terms = max(n_terms, out.helm.n_terms)

        # Order (b, t) to match the reshape used above.
        return Evaluation(
            actions=actions,
            reward=torch.stack(rewards, dim=1).reshape(-1),
            cost=torch.stack(costs, dim=1).reshape(-1),
            z=torch.stack(zs, dim=1).reshape(B * L, -1),
            converged=torch.stack(conv, dim=1).reshape(-1),
            n_terms=n_terms,
        )

    def loss(self, ev: Evaluation) -> torch.Tensor:
        """Eq. (19): -mean(r) + sum_n lambda_n mean(z_n)."""

        return -ev.reward.mean() + (self.lambdas * ev.z.mean(dim=0)).sum()

    def update_lambdas(self, z_mean: torch.Tensor) -> None:
        """Eq. (20)."""

        cfg = self.config
        self.lambdas = torch.clamp(
            self.lambdas + cfg.eta_lambda * z_mean.detach(),
            0.0,
            cfg.lambda_max,
        )

    def train_step(self, states: np.ndarray | torch.Tensor) -> dict:
        """
        One policy update on a minibatch of state windows, followed by
        the multiplier update.  Returns scalar metrics for logging.
        """

        states = torch.as_tensor(states, dtype=torch.float32)

        self.policy.train()
        ev = self.evaluate(states)
        loss = self.loss(ev)

        self.optimizer.zero_grad()
        loss.backward()

        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.grad_clip
            )

        self.optimizer.step()

        z_mean = ev.z.mean(dim=0).detach()
        self.update_lambdas(z_mean)

        return metrics(ev, loss=loss.item(), lambdas=self.lambdas,
                       names=self.critic.constraint_names)

    @torch.no_grad()
    def evaluate_metrics(self, states) -> dict:
        """Metrics on a batch without updating anything."""

        states = torch.as_tensor(states, dtype=torch.float32)
        self.policy.eval()
        ev = self.evaluate(states)
        return metrics(ev, loss=self.loss(ev).item(), lambdas=self.lambdas,
                       names=self.critic.constraint_names)


def metrics(ev: Evaluation, loss: float, lambdas, names) -> dict:
    """Scalar summary of an Evaluation."""

    z = ev.z.detach()
    total = z.sum(dim=1)

    m = {
        "loss": loss,
        "reward": ev.reward.detach().mean().item(),
        "cost": ev.cost.detach().mean().item(),
        "violation": total.mean().item(),
        "feasible": (total <= 1e-6).double().mean().item(),
        "helm_converged": ev.converged.double().mean().item(),
        "helm_terms": ev.n_terms,
    }
    for k, name in enumerate(names):
        m[f"z/{name}"] = z[:, k].mean().item()
        m[f"lambda/{name}"] = float(lambdas[k])
    return m
