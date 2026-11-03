"""
Assemble the D-HELM agent for any test system in environment/cases.py:
network, action space, D-HELM, critic, policy, state sampler, trainer.

One function so that training, evaluation and tests build exactly the
same setup.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandapower as pp
import torch

from dhelm_opf.algorithms.dhelm_cpo import CPOConfig, DHelmCPO
from dhelm_opf.environment.cases import CaseSpec, get_case
from dhelm_opf.environment.state_sampler import StateSampler
from dhelm_opf.environment.uncertainty import UncertaintyConfig
from dhelm_opf.helm.differentiable import DHelmNetwork
from dhelm_opf.helm.solver import HelmSolverConfig
from dhelm_opf.models.action_config import (
    ActionConfig,
    RampConfig,
    action_config_from_network,
)
from dhelm_opf.models.dhelm_critic import CriticConfig, DHELMCritic
from dhelm_opf.models.policy import DHelmPolicy


@dataclass
class Agent:
    case: CaseSpec
    net: pp.pandapowerNet
    action_config: ActionConfig
    uncertainty: UncertaintyConfig
    dhelm: DHelmNetwork
    critic: DHELMCritic
    policy: DHelmPolicy
    sampler: StateSampler
    cpo: DHelmCPO


def uncertainty_for(case: CaseSpec) -> UncertaintyConfig:
    return UncertaintyConfig(B_w=case.B_w, p_w_rated_mw=case.p_w_rated_mw)


def make_agent(
    case: str | CaseSpec = "case5",
    seed: int = 0,
    cpo_config: CPOConfig | None = None,
    critic_config: CriticConfig | None = None,
    solver_config: HelmSolverConfig | None = None,
    hidden: int | None = None,
    p_first: bool = True,
    ramp: RampConfig | None = None,
) -> Agent:
    """
    case:    name in environment/cases.py or a CaseSpec.
    hidden:  policy width; default from the case (Table 1).
    p_first: action order of Eq. (4) (p_g first).  False reproduces the
             voltage-first layout of runs made before that change.
    ramp:    Eq. (3) ramp limits; None disables ramping.

    cpo_config defaults to the case's Table 1 minibatch and lr.
    """

    spec = get_case(case) if isinstance(case, str) else case
    torch.manual_seed(seed)

    if cpo_config is None:
        cpo_config = CPOConfig(lr=spec.lr, batch_size=spec.batch_size)

    net = spec.build()
    action_config = action_config_from_network(net, p_first=p_first)
    uncertainty = uncertainty_for(spec)

    dhelm = DHelmNetwork(net, action_config, uncertainty, solver_config,
                         ramp=ramp)
    critic = DHELMCritic(dhelm, net, critic_config)

    sampler = StateSampler(dhelm.n_d, dhelm.n_w, uncertainty, seed=seed)

    policy = DHelmPolicy(
        state_dim=dhelm.state_dim,
        n_v=action_config.n_v,
        n_g=action_config.n_g,
        state_low=sampler.low,
        state_high=sampler.high,
        hidden=hidden or spec.hidden,
        idx_p=action_config.generator_action_indices,
    )

    cpo = DHelmCPO(policy, critic, cpo_config)

    return Agent(
        case=spec,
        net=net,
        action_config=action_config,
        uncertainty=uncertainty,
        dhelm=dhelm,
        critic=critic,
        policy=policy,
        sampler=sampler,
        cpo=cpo,
    )
