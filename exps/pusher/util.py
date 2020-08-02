"""Utilities for Pusher experiments."""

import numpy as np
import torch
import torch.distributions
import torch.nn as nn
from rllib.dataset.transforms import (
    ActionScaler,
    DeltaState,
    MeanFunction,
    NextStateClamper,
)
from rllib.environment import GymEnvironment

from exps.util import get_mb_mpo_agent, get_mpc_agent
from hucrl.reward.mujoco_rewards import PusherReward


class QuaternionTransform(nn.Module):
    """Transform pusher states to quaternion representation."""

    extra_dim = 7

    def forward(self, states):
        """Transform state before applying function approximation."""
        angles = states[..., :7]
        vel, obj = states[..., 7:14], states[..., 14:17]
        return torch.cat((torch.cos(angles), torch.sin(angles), vel, obj), dim=-1)
        # return states

    def inverse(self, states):
        """Inverse transformation of states."""
        cos, sin, other = states[..., :7], states[..., 7:14], states[..., 14:]
        angles = torch.atan2(sin, cos)
        return torch.cat((angles, other), dim=-1)


def large_state_termination(state, action, next_state=None):
    """Termination condition for environment."""
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state)
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action)

    return (state[..., -3:].abs() > 25).any(-1) | (state[..., 7:14].abs() > 2000).any(
        -1
    )


def get_agent_and_environment(params, agent_name):
    """Get experiment agent and environment."""
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    torch.set_num_threads(params.num_threads)

    # %% Define Environment.
    environment = GymEnvironment(
        "MBRLPusher-v0", action_cost=params.action_cost, seed=params.seed
    )
    action_scale = environment.action_scale
    reward_model = PusherReward(action_cost=params.action_cost)

    # %% Define Helper modules
    x_limits = (
        [-2.2854, -0.5236, -1.5, -2.3213, -1.5, -1.094, -1.5],
        [1.0, 1.3963, 1.7, 0, 1.5, 0, 1.5],
    )
    v_limits = (
        [-1e1, -1e1, -1e1, -1e1, -1e1, -1e1, -1e1],
        [+1e1, +1e1, +1e1, +1e1, +1e1, +1e1, +1e1],
    )

    obj_limits = ([0.3, -0.7, -0.275], [0.8, 0.1, -0.275])

    low = torch.tensor(x_limits[0] + v_limits[0] + obj_limits[0])
    high = torch.tensor(x_limits[1] + v_limits[1] + obj_limits[1])

    transformations = [
        NextStateClamper(low, high),
        ActionScaler(scale=action_scale),
        MeanFunction(DeltaState()),
    ]

    input_transform = QuaternionTransform()
    # input_transform = None
    x0 = (
        [-0.3, +0.3, -1.5, -1.5, +0.5, -1.094, -1.5],
        [+0.3, +0.6, -1.0, -1.0, +0.9, +0.000, +1.5],
    )
    v0 = (
        [-0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005],
        [+0.005, +0.005, +0.005, +0.005, +0.005, +0.005, +0.005],
    )

    obj0 = ([0.5, -0.4, -0.275], [0.7, -0.2, -0.275])

    exploratory_distribution = torch.distributions.Uniform(
        torch.tensor(x0[0] + v0[0] + obj0[0]), torch.tensor(x0[1] + v0[1] + obj0[1])
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
            termination=large_state_termination,
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
            termination=large_state_termination,
            initial_distribution=exploratory_distribution,
        )
    else:
        raise NotImplementedError

    return environment, agent
