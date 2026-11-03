"""
Backward-compatible 5-bus entry point.  The agent is now assembled for
any test system by algorithms/setup.py (make_agent).
"""

from __future__ import annotations

from dhelm_opf.algorithms.dhelm_cpo import CPOConfig
from dhelm_opf.algorithms.setup import Agent, make_agent
from dhelm_opf.helm.solver import HelmSolverConfig
from dhelm_opf.models.action_config import RampConfig
from dhelm_opf.models.dhelm_critic import CriticConfig

Case5Agent = Agent


def make_case5_agent(
    seed: int = 0,
    cpo_config: CPOConfig | None = None,
    critic_config: CriticConfig | None = None,
    solver_config: HelmSolverConfig | None = None,
    hidden: int = 64,
    p_first: bool = True,
    ramp: RampConfig | None = None,
) -> Agent:
    """The IEEE 5-bus agent; see make_agent."""

    return make_agent(
        "case5", seed=seed, cpo_config=cpo_config or CPOConfig(),
        critic_config=critic_config, solver_config=solver_config,
        hidden=hidden, p_first=p_first, ramp=ramp,
    )
