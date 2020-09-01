"""Utilities for Reacher experiments."""

import numpy as np
import torch
import torch.distributions
import torch.nn as nn
from rllib.dataset.transforms import (
    ActionScaler,
    AngleWrapper,
    DeltaState,
    MeanFunction,
    NextStateClamper,
)
from rllib.environment import GymEnvironment

from exps.util import LargeStateTermination, get_mb_mpo_agent, get_mpc_agent
from hucrl.reward.mujoco_rewards import ReacherReward


class QuaternionTransform(nn.Module):
    """Transform reacher states to quaternion representation."""

    extra_dim = 7

    def forward(self, states):
        """Transform state before applying function approximation."""
        angles, other = states[..., :7], states[..., 7:]
        return torch.cat((torch.cos(angles), torch.sin(angles), other), dim=-1)

    def inverse(self, states):
        """Inverse transformation of states."""
        cos, sin, other = states[..., :7], states[..., 7:14], states[..., 14:]
        angles = torch.atan2(sin, cos)
        return torch.cat((angles, other), dim=-1)


# def large_state_termination(state, action, next_state=None):
#     """Termination condition for environment."""
#     if not isinstance(state, torch.Tensor):
#         state = torch.tensor(state)
#     if not isinstance(action, torch.Tensor):
#         action = torch.tensor(action)
#
#     return torch.any(torch.abs(state) > 1e2, dim=-1) | torch.any(
#         torch.abs(action) > 25, dim=-1
#     )


def get_agent_and_environment(params, agent_name):
    """Get experiment agent and environment."""
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    torch.set_num_threads(params.num_threads)

    # %% Define Environment.
    environment = GymEnvironment(
        "MBRLReacher3D-v0", action_cost=params.action_cost, seed=params.seed
    )
    action_scale = environment.action_scale
    reward_model = ReacherReward(action_cost=params.action_cost)

    # %% Define Helper modules
    low = torch.tensor(
        [
            -2.2854,
            -0.5236,
            -3.9,
            -2.3213,
            -1e1,
            -2.094,
            -1e1,
            -1e2,
            -1e2,
            -1e2,
            -1e2,
            -1e2,
            -1e2,
            -1e2,
            # -0.4,
            # -0.1,
            # -0.4,
        ]
    )

    high = torch.tensor(
        [
            1.714602,
            1.3963,
            0.8,
            0,
            1e1,
            0,
            1e1,
            1e2,
            1e2,
            1e2,
            1e2,
            1e2,
            1e2,
            1e2,
            # 0.4,
            # 0.6,
            # 0.4,
        ]
    )

    transformations = [
        ActionScaler(scale=action_scale),
        AngleWrapper(indexes=[0, 1, 2, 3, 4, 5, 6]),
        MeanFunction(DeltaState()),  #
        NextStateClamper(low, high),  # , constant_idx=[14, 15, 16]),
    ]

    input_transform = QuaternionTransform()
    # input_transform = None
    exploratory_distribution = torch.distributions.Uniform(
        torch.tensor(
            [
                0.3,
                -0.3,
                -1.5,
                -2.5,
                -1.5,
                -2.0,
                -1.5,
                -0.005,
                -0.005,
                -0.005,
                -0.005,
                -0.005,
                -0.005,
                -0.005,  # qvel
                # -0.3,
                # -0.1,
                # -0.3,  # goal
            ]
        ),
        torch.tensor(
            [
                0.8,
                0.3,
                1.5,
                -1.5,
                1.5,
                0,
                1.5,  # qpos
                0.005,
                0.005,
                0.005,
                0.005,
                0.005,
                0.005,
                0.005,  # qvel
                # 0.3,
                # 0.6,
                # 0.3,  # goal
            ]
        ),
    )

    if agent_name == "mpc":
        agent = get_mpc_agent(
            environment.dim_state,
            # (environment.dim_state[0] + 3,),
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
            environment.dim_state,  # (environment.dim_state[0] + 3,),
            environment.dim_action,
            params=params,
            reward_model=reward_model,
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
