"""Utilities for cartpole experiments."""

import numpy as np
import torch
import torch.distributions
import torch.nn as nn
from rllib.dataset.transforms import ActionScaler, DeltaState, MeanFunction
from rllib.environment import GymEnvironment

from exps.util import LargeStateTermination, get_mb_mpo_agent, get_mpc_agent
from hucrl.reward.mujoco_rewards import CartPoleReward


class StateTransform(nn.Module):
    """Transform pendulum states to cos, sin, angular_velocity."""

    extra_dim = 1

    def forward(self, states_):
        """Transform state before applying function approximation."""
        position, angle, velocity, angular_velocity = torch.split(states_, 1, dim=-1)
        states_ = torch.cat(
            (torch.cos(angle), torch.sin(angle), position, velocity, angular_velocity),
            dim=-1,
        )
        return states_

    def inverse(self, states_):
        """Inverse transformation of states."""
        cos, sin, position, velocity, angular_velocity = torch.split(states_, 1, dim=-1)
        angle = torch.atan2(sin, cos)
        states_ = torch.cat((position, angle, velocity, angular_velocity), dim=-1)
        return states_


def get_agent_and_environment(params, agent_name):
    """Get experiment agent and environment."""
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    torch.set_num_threads(params.num_threads)

    # %% Define Environment.
    environment = GymEnvironment(
        "MBRLCartPole-v0", action_cost=params.action_cost, seed=params.seed
    )
    action_scale = environment.action_scale

    reward_model = CartPoleReward(action_cost=params.action_cost)
    # %% Define Helper modules
    transformations = [
        ActionScaler(scale=action_scale),
        MeanFunction(DeltaState()),  # AngleWrapper(indexes=[1])
    ]

    input_transform = StateTransform()
    exploratory_distribution = torch.distributions.Uniform(
        torch.tensor([-1.25, -np.pi, -0.05, -0.05]),
        torch.tensor([+1.25, +np.pi, +0.05, +0.05]),
    )

    if agent_name == "mpc":
        agent = get_mpc_agent(
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            params=params,
            reward_model=reward_model,
            action_scale=action_scale,
            transformations=transformations,
            input_transform=input_transform,
            termination_model=LargeStateTermination(),
            initial_distribution=exploratory_distribution,
        )
    elif agent_name == "mbmpo":
        agent = get_mb_mpo_agent(
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            params=params,
            reward_model=reward_model,
            input_transform=input_transform,
            action_scale=action_scale,
            transformations=transformations,
            termination_model=LargeStateTermination(),
            initial_distribution=exploratory_distribution,
        )
    else:
        raise NotImplementedError

    return environment, agent
