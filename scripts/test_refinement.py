"""
Tests for R1 / R2 action refinement and the cached OPF reference.

Actions come from a briefly trained policy perturbed with noise, so
the test set has both feasible and infeasible states.

1. Refined actions stay inside [-1, 1].
2. R1 leaves feasible states untouched and never increases the total
   violation of an infeasible one; it restores feasibility for most.
3. R2 is never worse than its start (lexicographic: feasible first,
   then cost) and lowers the mean cost of feasible states.
4. The OPF cache reproduces a fresh solve, and the OPF solutions are
   feasible for the critic with the same cost.

Run:
    python scripts/test_refinement.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from dhelm_opf.algorithms.case5_setup import make_case5_agent
from dhelm_opf.algorithms.refine import RefineConfig, refine_actions
from dhelm_opf.utils.evaluation import check_opf_reference
from dhelm_opf.utils.opf_reference import compute_reference, load_or_compute

TOL = 1e-6


def setup(n=64, train_steps=80, noise=0.25):
    agent = make_case5_agent(seed=3)
    for _ in range(train_steps):
        agent.cpo.train_step(agent.sampler.sample(32))

    states = agent.sampler.sample(n)
    with torch.no_grad():
        a = agent.policy(torch.as_tensor(states[:, None], dtype=torch.float32))
    a = a.squeeze(1).double()
    gen = torch.Generator().manual_seed(0)
    a = (a + noise * torch.randn(a.shape, generator=gen, dtype=torch.float64)).clamp(-1, 1)

    with torch.no_grad():
        out = agent.critic(torch.as_tensor(states), a)
    return agent, torch.as_tensor(states), a, out


def test_r1(agent, states, a0, out0) -> None:

    viol0 = out0.z.sum(dim=1)
    feas0 = viol0 <= TOL
    assert feas0.any() and (~feas0).any(), "test set needs both kinds"

    r = refine_actions(agent.critic, agent.cpo.lam, states, a0,
                       RefineConfig(mode="r1", K=50))
    viol = r.z.sum(dim=1)

    assert torch.all(r.a_hat.abs() <= 1.0)
    assert torch.equal(r.refined, ~feas0)
    assert torch.equal(r.a_hat[feas0], a0[feas0])          # untouched
    assert torch.all(viol <= viol0 + 1e-12)                  # never worse

    fixed = (viol[~feas0] <= TOL).double().mean().item()
    assert fixed >= 0.8, fixed

    print(f"  1-2. R1: {int((~feas0).sum())} infeasible states refined, "
          f"{fixed:.0%} made feasible; feasible states untouched; "
          f"violation never increased")


def test_r2(agent, states, a0, out0) -> None:

    cost0 = out0.f_sum
    viol0 = out0.z.sum(dim=1)
    feas0 = viol0 <= TOL

    r = refine_actions(agent.critic, agent.cpo.lam, states, a0,
                       RefineConfig(mode="r2", K=50))
    viol = r.z.sum(dim=1)

    assert torch.all(r.a_hat.abs() <= 1.0)
    assert bool(r.refined.all())

    # Feasible states stay feasible and do not get more expensive.
    assert bool((viol[feas0] <= TOL).all())
    assert torch.all(r.f_sum[feas0] <= cost0[feas0] + 1e-9)
    # Infeasible states: violation never increases.
    assert torch.all(viol[~feas0] <= viol0[~feas0] + 1e-12)

    saving = (1 - r.f_sum[feas0].mean() / cost0[feas0].mean()).item()
    assert saving > 0.0, saving

    print(f"  3. R2: all {len(states)} states refined; feasible states "
          f"{saving:.1%} cheaper on average; never worse than the start; "
          f"feasible share {feas0.double().mean():.0%} -> "
          f"{(viol <= TOL).double().mean():.0%}")


def test_opf_cache(agent) -> None:

    states = agent.sampler.sample(4)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ref.npz"
        first = load_or_compute(path, states, agent.uncertainty,
                                agent.action_config, verbose=False)
        cached = load_or_compute(path, states, agent.uncertainty,
                                 agent.action_config, verbose=False)
        fresh = compute_reference(states, agent.uncertainty, agent.action_config)

    assert np.array_equal(first["cost"], cached["cost"])
    assert np.allclose(cached["cost"], fresh["cost"], rtol=1e-8)

    check = check_opf_reference(agent, cached)
    assert check["opf_converged"] == 4
    assert check["cost_mismatch_max"] < 0.1, check
    assert check["violation_max"] < 1e-5, check

    print(f"  4. OPF cache = fresh solve; OPF@critic cost mismatch "
          f"{check['cost_mismatch_max']:.1e} $/h, violation "
          f"{check['violation_max']:.1e}")


def main() -> None:

    torch.manual_seed(0)
    print("Refinement / OPF-cache tests")

    agent, states, a0, out0 = setup()
    test_r1(agent, states, a0, out0)
    test_r2(agent, states, a0, out0)
    test_opf_cache(agent)

    print("All refinement tests passed.")


if __name__ == "__main__":
    main()
