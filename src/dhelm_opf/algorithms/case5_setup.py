"""
Assemble the 5-bus D-HELM agent: network, action space, D-HELM,
critic, policy, state sampler and trainer.

One function so that training, evaluation and tests build exactly the
same setup.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandapower as pp
import torch

from dhelm_opf.algorithms.dhelm_cpo import CPOConfig, DHelmCPO
from dhelm_opf.environment.case5 import build_case5
from dhelm_opf.environment.state_sampler import StateSampler
from dhelm_opf.environment.uncertainty import UncertaintyConfig
from dhelm_opf.helm.differentiable import DHelmNetwork
from dhelm_opf.helm.solver import HelmSolverConfig
from dhelm_opf.models.action_config import ActionConfig, action_config_from_network
from dhelm_opf.models.dhelm_critic import CriticConfig, DHELMCritic
from dhelm_opf.models.policy import DHelmPolicy


@dataclass
class Case5Agent:
    net: pp.pandapowerNet
    action_config: ActionConfig
    uncertainty: UncertaintyConfig
    dhelm: DHelmNetwork
    critic: DHELMCritic
    policy: DHelmPolicy
    sampler: StateSampler
    cpo: DHelmCPO


def make_case5_agent(
    seed: int = 0,
    cpo_config: CPOConfig | None = None,
    critic_config: CriticConfig | None = None,
    solver_config: HelmSolverConfig | None = None,
    hidden: int = 64,
) -> Case5Agent:

    torch.manual_seed(seed)

    net = build_case5()
    action_config = action_config_from_network(net)
    uncertainty = UncertaintyConfig()

    dhelm = DHelmNetwork(net, action_config, uncertainty, solver_config)
    critic = DHELMCritic(dhelm, net, critic_config)

    sampler = StateSampler(dhelm.n_d, dhelm.n_w, uncertainty, seed=seed)

    policy = DHelmPolicy(
        state_dim=dhelm.state_dim,
        n_v=action_config.n_v,
        n_g=action_config.n_g,
        state_low=sampler.low,
        state_high=sampler.high,
        hidden=hidden,
    )

    cpo = DHelmCPO(policy, critic, cpo_config)

    return Case5Agent(
        net=net,
        action_config=action_config,
        uncertainty=uncertainty,
        dhelm=dhelm,
        critic=critic,
        policy=policy,
        sampler=sampler,
        cpo=cpo,
    )
