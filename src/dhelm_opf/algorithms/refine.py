"""
Post-training action refinement through the D-HELM critic (Remark 5).

Paper basis (Remark 5)
----------------------
"The remaining constraints ... are addressed using an online
feasibility test to retain only feasible solutions.  In cases where
infeasibility arises, a small number of gradient steps are performed to
adjust control actions without modifying all policy parameters ...
the D-HELM critic model refines its inputs using established physical
layers."

Both variants take gradient steps on the ACTION of each state (the
policy weights are untouched), on the per-state Lagrangian

    J(a) = -r(a) + sum_n lambda_n z_n(a)

with the multipliers lambda learned during training.

    R1 (paper-faithful)
        Only states that FAIL the feasibility test are refined, and a
        state stops as soon as it becomes feasible.  Feasible states are
        returned unchanged.

    R2 (our extension; not described in the paper)
        EVERY state is refined for the full step budget, so the cost is
        lowered as well.  The policy then acts as a warm start for a
        first-order OPF solve through the critic.

Implementation
--------------
* The optimization variable is the pre-tanh action u, a = tanh(u), so
  actions can never leave their bounds (as in Eq. 18).
* All states are refined in one batch.  The loss is the SUM of J over
  the active states and Adam is element-wise, so each state's update is
  exactly what it would be if it were refined alone.
* Best-iterate tracking per state, with the starting action as a
  candidate.  Order: feasible beats infeasible; among feasible, lower
  cost wins; among infeasible, lower total violation wins.  Iterates at
  which the D-HELM did not converge are never accepted.  Refinement can
  therefore never make a state worse.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from dhelm_opf.models.dhelm_critic import DHELMCritic

DTYPE = torch.float64


@dataclass(frozen=True)
class RefineConfig:
    mode: str = "r1"            # "r1" (paper) or "r2" (extension)
    steps: int = 20             # gradient-step budget K
    lr: float = 0.05            # Adam step size in pre-tanh space
    feasible_tol: float = 1e-6  # feasibility test: sum(z) <= tol

    def __post_init__(self):
        if self.mode not in ("r1", "r2"):
            raise ValueError("mode must be 'r1' or 'r2'.")
        if self.steps < 0:
            raise ValueError("steps must be >= 0.")


@dataclass
class RefineResult:
    actions: torch.Tensor       # (S, a) refined normalized actions
    cost: torch.Tensor          # (S,)
    z: torch.Tensor             # (S, n_constraint)
    feasible: torch.Tensor      # (S,) bool
    refined: torch.Tensor       # (S,) bool, state was optimized at all
    steps_taken: torch.Tensor   # (S,) gradient steps applied per state
    seconds: float


def refine_actions(
    critic: DHELMCritic,
    lambdas: torch.Tensor,
    states: torch.Tensor,
    actions: torch.Tensor,
    config: RefineConfig,
) -> RefineResult:
    """Refine (S, a) actions for (S, d) states.  See module docstring."""

    ccfg = critic.config
    if ccfg.ramp_up_mw is not None or ccfg.ramp_down_mw is not None:
        raise NotImplementedError(
            "Refinement with ramping needs the previous step's slack "
            "output; not implemented."
        )

    start = time.perf_counter()

    states = torch.as_tensor(states, dtype=DTYPE)
    a0 = torch.as_tensor(actions, dtype=DTYPE).detach()
    lam = torch.as_tensor(lambdas, dtype=DTYPE)
    tol = config.feasible_tol

    eps = 1e-9
    u = torch.atanh(a0.clamp(-1 + eps, 1 - eps)).clone().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=config.lr)

    S = states.shape[0]

    best_a = a0.clone()
    best_cost = torch.full((S,), float("inf"), dtype=DTYPE)
    best_viol = torch.full((S,), float("inf"), dtype=DTYPE)
    best_feas = torch.zeros(S, dtype=torch.bool)

    active = None
    refined = torch.zeros(S, dtype=torch.bool)
    steps_taken = torch.zeros(S, dtype=torch.long)

    for k in range(config.steps + 1):

        a = torch.tanh(u)
        out = critic(states, a)

        with torch.no_grad():
            cost = out.cost.detach()
            viol = out.z.detach().sum(dim=1)
            feas = viol <= tol
            valid = out.helm.converged

            # Best-iterate update (lexicographic, see module docstring).
            better = valid & (
                (feas & ~best_feas)
                | (feas & best_feas & (cost < best_cost))
                | (~feas & ~best_feas & (viol < best_viol))
            )
            # At k = 0 the candidate is the ORIGINAL action, not
            # tanh(atanh(a0)), so untouched states return a0 exactly.
            cand = a0 if k == 0 else a.detach()
            best_a[better] = cand[better]
            best_cost[better] = cost[better]
            best_viol[better] = viol[better]
            best_feas[better] = feas[better]

            if k == 0:
                active = ~feas if config.mode == "r1" else torch.ones(S, dtype=torch.bool)
                refined = active.clone()
            elif config.mode == "r1":
                # R1: a state stops as soon as it is feasible.
                active = active & ~(feas & valid)

        if k == config.steps or not bool(active.any()):
            break

        J = -out.reward + (lam * out.z).sum(dim=1)
        loss = (J * active.to(DTYPE)).sum()

        opt.zero_grad()
        loss.backward()

        frozen = u.detach().clone()
        opt.step()
        with torch.no_grad():
            u[~active] = frozen[~active]      # Adam momentum must not move them

        steps_taken += active.long()

    with torch.no_grad():
        final = critic(states, best_a)

    viol = final.z.sum(dim=1)

    return RefineResult(
        actions=best_a,
        cost=final.cost.detach(),
        z=final.z.detach(),
        feasible=viol <= tol,
        refined=refined,
        steps_taken=steps_taken,
        seconds=time.perf_counter() - start,
    )
