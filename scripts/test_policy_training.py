"""
Unit tests for the policy, the states-only buffer and the trainer.

1. Policy shapes and output range; autoregressive feedback of the
   previous generator action.
2. Buffer windows come from consecutive states of one episode.
3. Gradient reaches every policy parameter through the D-HELM critic.
4. Eq. (20) multiplier update (step and clipping).
5. Smoke run: a short training run lowers the loss on a fixed batch.

Run:
    python scripts/test_policy_training.py
"""

from __future__ import annotations

import numpy as np
import torch

from dhelm_opf.algorithms.case5_setup import make_case5_agent
from dhelm_opf.algorithms.dhelm_cpo import CPOConfig
from dhelm_opf.environment.state_sampler import StateBuffer


def test_policy(agent) -> None:

    policy = agent.policy
    states = torch.as_tensor(agent.sampler.sample(6 * 5).reshape(6, 5, -1),
                             dtype=torch.float32)

    actions = policy(states)
    assert actions.shape == (6, 5, 8)
    assert torch.all(actions.abs() <= 1.0)
    assert policy.input_dim == 12          # Table 1: 8 + 4

    # Step t must use the generator part of the action at t-1.
    a0, hc = policy.step(states[:, 0], torch.zeros(6, 4))
    a1, _ = policy.step(states[:, 1], a0[:, 4:], hc)
    assert torch.allclose(a1, actions[:, 1], atol=1e-6)

    print("  1. policy shapes / range / feedback : OK (input 12, output 8)")


def test_buffer(agent) -> None:

    buf = StateBuffer()
    for _ in range(3):
        buf.add_episode(agent.sampler.episode(25))

    rng = np.random.default_rng(0)
    w = buf.sample_windows(16, 5, rng)
    assert w.shape == (16, 5, 8)

    data = np.stack(buf._episodes)
    for window in w:
        hits = [
            (e, s)
            for e in range(len(data))
            for s in range(25 - 5 + 1)
            if np.array_equal(data[e, s:s + 5], window)
        ]
        assert len(hits) == 1

    print("  2. buffer windows are consecutive   : OK")


def test_gradient_flow(agent) -> None:

    cpo = agent.cpo

    # A window of >= 2 steps is needed: with L = 1 the LSTM's hidden
    # state starts at zero, so weight_hh correctly gets no gradient.
    states = torch.as_tensor(
        agent.sampler.sample(32 * 3).reshape(32, 3, -1), dtype=torch.float32
    )

    cpo.policy.zero_grad()
    loss = cpo.loss(cpo.evaluate(states))
    loss.backward()

    for name, p in cpo.policy.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name

    print("  3. gradient reaches all parameters  : OK "
          f"({sum(p.numel() for p in cpo.policy.parameters())} parameters)")


def test_lambda_update() -> None:

    agent = make_case5_agent(
        cpo_config=CPOConfig(eta_lambda=2.0, lambda_max=12.0)
    )
    cpo = agent.cpo
    n = len(cpo.lambdas)

    z = torch.linspace(0.0, 2.0, n, dtype=torch.float64)
    cpo.update_lambdas(z)

    expected = torch.clamp(10.0 + 2.0 * z, 0.0, 12.0)
    assert torch.allclose(cpo.lambdas, expected)

    print("  4. lambda update (Eq. 20)           : OK")


def test_smoke_training(steps: int = 60) -> None:

    agent = make_case5_agent(seed=1)
    cpo, sampler = agent.cpo, agent.sampler

    fixed = sampler.sample(128)
    lam0 = cpo.lambdas.clone()

    before = cpo.evaluate_metrics(fixed)
    for _ in range(steps):
        cpo.train_step(sampler.sample(32))

    # Compare at the initial multipliers so only the policy changed.
    cpo.lambdas = lam0
    after = cpo.evaluate_metrics(fixed)

    assert after["loss"] < before["loss"], (before["loss"], after["loss"])

    print(f"  5. smoke run ({steps} steps)            : loss "
          f"{before['loss']:.3f} -> {after['loss']:.3f}, "
          f"violation {before['violation']:.3f} -> {after['violation']:.3f}, "
          f"cost {before['cost']:.0f} -> {after['cost']:.0f} $/h")


def main() -> None:

    agent = make_case5_agent(seed=0)

    print("Policy / trainer tests")

    test_policy(agent)
    test_buffer(agent)
    test_gradient_flow(agent)
    test_lambda_update()
    test_smoke_training()

    print("All policy / trainer tests passed.")


if __name__ == "__main__":
    main()
