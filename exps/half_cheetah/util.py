"""Utilities for Half-Cheetah experiments."""

import numpy as np
import torch
import torch.distributions
import torch.nn as nn
from rllib.dataset.transforms import ActionScaler, DeltaState, MeanFunction
from rllib.environment import GymEnvironment

from exps.util import LargeStateTermination, get_mb_mpo_agent, get_mpc_agent
from hucrl.reward.mujoco_rewards import HalfCheetahReward


class StateTransform(nn.Module):
    """Transform pendulum states to cos, sin, angular_velocity."""

    extra_dim = 8

    def forward(self, states):
        """Transform state before applying function approximation."""
        angles = states[..., 1:9]
        states_ = torch.cat(
            (states[..., :1], torch.cos(angles), torch.sin(angles), states[..., 9:]),
            dim=-1,
        )
        return states_

    def inverse(self, states):
        """Inverse transformation of states."""
        cos, sin = states[..., 2:9], states[..., 9:16]
        angle = torch.atan2(sin, cos)
        states_ = torch.cat((states[..., :2], angle, states[..., 16:]), dim=-1)
        return states_


def get_agent_and_environment(params, agent_name):
    """Get experiment agent and environment."""
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    torch.set_num_threads(params.num_threads)

    # %% Define Environment.
    environment = GymEnvironment(
        "MBRLHalfCheetah-v0", action_cost=params.action_cost, seed=params.seed
    )
    action_scale = environment.action_scale
    reward_model = HalfCheetahReward(action_cost=params.action_cost)

    # %% Define Helper modules
    transformations = [
        ActionScaler(scale=action_scale),
        MeanFunction(DeltaState()),  # AngleWrapper(indexes=[1])
    ]

    input_transform = None
    exploratory_distribution = torch.distributions.MultivariateNormal(
        torch.zeros(environment.dim_state[0]),
        1e-6 * torch.eye(environment.dim_state[0]),
    )

    if agent_name == "mpc":
        agent = get_mpc_agent(
            environment.dim_state,
            environment.dim_action,
            params,
            reward_model,
            action_scale=action_scale,
            transformations=transformations,
            input_transform=input_transform,
            termination_model=LargeStateTermination(
                max_action=environment.action_scale.max() * 15
            ),
            initial_distribution=exploratory_distribution,
        )
    elif agent_name == "mbmpo":
        agent = get_mb_mpo_agent(
            environment.dim_state,
            environment.dim_action,
            params,
            reward_model,
            input_transform=input_transform,
            action_scale=action_scale,
            transformations=transformations,
            termination_model=LargeStateTermination(
                max_action=environment.action_scale.max() * 15
            ),
            initial_distribution=exploratory_distribution,
        )
    else:
        raise NotImplementedError

    return environment, agent
