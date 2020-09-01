import numpy as np
import pytest
import torch
from gym import envs
from rllib.environment import GymEnvironment

from hucrl.reward.mujoco_rewards import (
    CartPoleReward,
    HalfCheetahReward,
    HalfCheetahV2Reward,
    ReacherReward,
)


@pytest.fixture(params=[0, 0.1, None])
def action_cost(request):
    return request.param


@pytest.fixture(params=["zero", "random"])
def action_type(request):
    return request.param


@pytest.fixture(
    params=[
        ("MBRLCartPole-v0", CartPoleReward),
        ("MBRLHalfCheetah-v0", HalfCheetahReward),
        ("MBRLHalfCheetah-v2", HalfCheetahV2Reward),
        # ('MBRLPusher-v0', PusherReward),
        ("MBRLReacher3D-v0", ReacherReward),
    ]
)
def environment(request):
    return request.param


# @pytest.mark.xfail(raises=AttributeError, reason="Mujoco not installed.")
@pytest.mark.skipif(
    "MBRLCartPole-v0" not in envs.registry.env_specs.keys(),
    reason="Mujoco not installed.",
)
def test_reward(environment, action_cost, action_type):
    env_name, reward_model_ = environment
    if action_cost is not None:
        env = GymEnvironment(env_name, action_cost=action_cost)
    else:
        env = GymEnvironment(env_name)
    state = env.reset()
    if action_cost is not None:
        reward_model = reward_model_(action_cost=action_cost)
    else:
        reward_model = reward_model_()
    reward_model.set_goal(env.goal)
    for _ in range(50):
        if action_type == "random":
            action = env.action_space.sample()
        elif action_type == "zero":
            action = np.zeros(env.dim_action)
        else:
            raise NotImplementedError

        next_state, reward, done, info = env.step(action)
        if env.goal is not None:
            state = np.concatenate((state, env.goal))
        np.testing.assert_allclose(
            reward, reward_model(state, action, next_state)[0], rtol=1e-3, atol=1e-6
        )

        np.testing.assert_allclose(
            np.tile(reward, (5,)),
            reward_model(
                np.tile(state, (5, 1)),
                np.tile(action, (5, 1)),
                np.tile(next_state, (5, 1)),
            )[0],
            rtol=1e-3,
            atol=1e-6,
        )

        state = torch.tensor(state, dtype=torch.get_default_dtype())
        action = torch.tensor(action, dtype=torch.get_default_dtype())
        next_state = torch.tensor(next_state, dtype=torch.get_default_dtype())
        np.testing.assert_allclose(
            reward, reward_model(state, action, next_state)[0], rtol=1e-3, atol=1e-6
        )

        np.testing.assert_allclose(
            np.tile(reward, (5, 1)),
            reward_model(
                state.repeat(5, 1), action.repeat(5, 1), next_state.repeat(5, 1)
            )[0],
            rtol=1e-3,
            atol=1e-6,
        )

        state = next_state.numpy()


@pytest.mark.skipif(
    "MBRLReacher3D-v0" not in envs.registry.env_specs.keys(),
    reason="Mujoco not installed.",
)
def test_tolerance(action_cost):
    env_name, reward_model_ = ("MBRLReacher3D-v0", ReacherReward)
    if action_cost is not None:
        env = GymEnvironment(env_name, action_cost=action_cost, sparse=True)
    else:
        env = GymEnvironment(env_name, sparse=True)
    state = env.reset()
    if action_cost is not None:
        reward_model = reward_model_(action_cost=action_cost, sparse=True)
    else:
        reward_model = reward_model_(sparse=True)
    reward_model.set_goal(env.goal)

    for _ in range(50):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        if env.goal is not None:
            state = np.concatenate((state, env.goal))
        np.testing.assert_allclose(
            reward, reward_model(state, action, next_state)[0], rtol=1e-3, atol=1e-6
        )

        np.testing.assert_allclose(
            np.tile(reward, (5,)),
            reward_model(
                np.tile(state, (5, 1)),
                np.tile(action, (5, 1)),
                np.tile(next_state, (5, 1)),
            )[0],
            rtol=1e-3,
            atol=1e-6,
        )

        state = torch.tensor(state, dtype=torch.get_default_dtype())
        action = torch.tensor(action, dtype=torch.get_default_dtype())
        next_state = torch.tensor(next_state, dtype=torch.get_default_dtype())
        np.testing.assert_allclose(
            reward, reward_model(state, action, next_state)[0], rtol=1e-3, atol=1e-6
        )

        np.testing.assert_allclose(
            np.tile(reward, (5, 1)),
            reward_model(
                state.repeat(5, 1), action.repeat(5, 1), next_state.repeat(5, 1)
            )[0],
            rtol=1e-3,
            atol=1e-6,
        )

        state = next_state.numpy()
