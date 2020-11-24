"""Utilities for inverted pendulum experiments."""
import argparse
from typing import List

import gpytorch
import numpy as np
import torch
import torch.distributions
import torch.nn as nn
import torch.optim as optim
from rllib.dataset.transforms import ActionScaler, DeltaState, MeanFunction
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.environment.system_environment import SystemEnvironment
from rllib.environment.systems import InvertedPendulum
from rllib.model.abstract_model import AbstractModel
from rllib.policy import NNPolicy
from rllib.reward.utilities import tolerance
from rllib.util.neural_networks.utilities import freeze_parameters
from rllib.util.rollout import rollout_model, rollout_policy
from rllib.value_function import NNValueFunction
from torch.distributions import MultivariateNormal
from tqdm import tqdm

from exps.inverted_pendulum.plotters import (
    plot_learning_losses,
    plot_returns_entropy_kl,
    plot_trajectory_states_and_rewards,
    plot_values_and_policy,
)
from exps.util import get_mb_mpo_agent, get_mpc_agent
from hucrl.algorithms.mbmpo import MBMPO
from hucrl.environment.hallucination_wrapper import HallucinationWrapper


class StateTransform(nn.Module):
    """Transform pendulum states to cos, sin, angular_velocity."""

    extra_dim = 1

    def forward(self, states_):
        """Transform state before applying function approximation."""
        angle, angular_velocity = torch.split(states_, 1, dim=-1)
        states_ = torch.cat(
            (torch.cos(angle), torch.sin(angle), angular_velocity), dim=-1
        )
        return states_

    def inverse(self, states_):
        """Inverse transformation of states."""
        cos, sin, angular_velocity = torch.split(states_, 1, dim=-1)
        angle = torch.atan2(sin, cos)
        states_ = torch.cat((angle, angular_velocity), dim=-1)
        return states_


def large_state_termination(state, action, next_state=None):
    """Termination condition for environment."""
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state)
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action)

    done = torch.any(torch.abs(state) > 200, dim=-1) | torch.any(
        torch.abs(action) > 200, dim=-1
    )
    return (
        torch.zeros(*done.shape, 2)
        .scatter_(dim=-1, index=(~done).long().unsqueeze(-1), value=-float("inf"))
        .squeeze(-1)
    )


class PendulumReward(AbstractModel):
    """Reward for Inverted Pendulum."""

    def __init__(self, action_cost=0):
        super().__init__(dim_state=(2,), dim_action=(1,), model_kind="rewards")
        self.action_cost = action_cost
        self.reward_offset = 0

    def forward(self, state, action, next_state):
        """See `abstract_reward.forward'."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.get_default_dtype())
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.get_default_dtype())

        cos_angle = torch.cos(state[..., 0])
        velocity = state[..., 1]

        angle_tolerance = tolerance(cos_angle, lower=0.95, upper=1.0, margin=0.1)
        velocity_tolerance = tolerance(velocity, lower=-0.5, upper=0.5, margin=0.5)
        state_cost = angle_tolerance * velocity_tolerance

        action_tolerance = tolerance(action[..., 0], lower=-0.1, upper=0.1, margin=0.1)
        action_cost = self.action_cost * (action_tolerance - 1)

        cost = state_cost + action_cost

        return cost.unsqueeze(-1), torch.zeros(1)


class PendulumModel(AbstractModel):
    """Pendulum Model.

    Torch implementation of a pendulum model using euler forwards integration.
    """

    def __init__(
        self, mass, length, friction, step_size=1 / 80, noise: MultivariateNormal = None
    ):
        super().__init__(dim_state=(2,), dim_action=(1,))
        self.mass = mass
        self.length = length
        self.friction = friction
        self.step_size = step_size
        self.noise = noise

    def forward(self, state, action):
        """Get next-state distribution."""
        # Physical dynamics
        action = torch.clamp(action, -1.0, 1.0)
        mass = self.mass
        gravity = 9.81
        length = self.length
        friction = self.friction
        inertia = mass * length ** 2
        dt = self.step_size

        angle, angular_velocity = torch.split(state, 1, dim=-1)
        for _ in range(1):
            x_ddot = (
                (gravity / length) * torch.sin(angle)
                + action * (1 / inertia)
                - (friction / inertia) * angular_velocity
            )

            angle = angle + dt * angular_velocity
            angular_velocity = angular_velocity + dt * x_ddot

        next_state = torch.cat((angle, angular_velocity), dim=-1)

        if self.noise is None:
            return next_state, torch.zeros(1)
        else:
            return next_state + self.noise.mean, self.noise.covariance_matrix


def test_policy_on_model(
    dynamical_model, reward_model, policy, test_state, policy_str="Sampled Policy"
):
    """Test a policy on a model."""
    with torch.no_grad():
        trajectory = rollout_model(
            dynamical_model,
            reward_model,
            policy,
            max_steps=400,
            initial_state=test_state.unsqueeze(0).unsqueeze(1),
        )
        trajectory = stack_list_of_tuples(trajectory)

    states = trajectory.state[:, 0]
    rewards = trajectory.reward
    plot_trajectory_states_and_rewards(states, rewards)

    model_rewards = torch.sum(rewards).item()
    print(f"Model with {policy_str} Cumulative reward: {model_rewards:.2f}")

    return model_rewards, trajectory


def test_policy_on_environment(
    environment, policy, test_state, policy_str="Sampled Policy"
):
    """Test a policy on an environment."""
    environment.state = test_state.numpy()
    environment.initial_state = lambda: test_state.numpy()
    trajectory = rollout_policy(environment, policy, max_steps=400, render=False)[0]

    trajectory = stack_list_of_tuples(trajectory)
    env_rewards = torch.sum(trajectory.reward).item()
    print(f"Environment with {policy_str} Cumulative reward: {env_rewards:.2f}")

    return env_rewards, trajectory


def train_mpo(
    mpo: MBMPO,
    initial_distribution,
    optimizer,
    num_iter,
    num_trajectories,
    num_simulation_steps,
    num_gradient_steps,
    batch_size,
    num_subsample,
):
    """Train MPO policy."""
    value_losses = []  # type: List[float]
    policy_losses = []  # type: List[float]
    returns = []  # type: List[float]
    kl_div = []  # type: List[float]
    entropy = []  # type: List[float]
    for i in tqdm(range(num_iter)):
        # Compute the state distribution
        state_batches = _simulate_model(
            mpo,
            initial_distribution,
            num_trajectories,
            num_simulation_steps,
            batch_size,
            num_subsample,
            returns,
            entropy,
        )

        policy_episode_loss, value_episode_loss, episode_kl_div = _optimize_policy(
            mpo, state_batches, optimizer, num_gradient_steps
        )

        value_losses.append(value_episode_loss / len(state_batches))
        policy_losses.append(policy_episode_loss / len(state_batches))
        kl_div.append(episode_kl_div)

    return value_losses, policy_losses, kl_div, returns, entropy


def _simulate_model(
    mpo,
    initial_distribution,
    num_trajectories,
    num_simulation_steps,
    batch_size,
    num_subsample,
    returns,
    entropy,
):
    with torch.no_grad():
        test_states = torch.tensor([np.pi, 0]).repeat(num_trajectories // 2, 1)
        initial_states = initial_distribution.sample((num_trajectories // 2,))
        initial_states = torch.cat((initial_states, test_states), dim=0)
        trajectory = rollout_model(
            mpo.dynamical_model,
            reward_model=mpo.reward_model,
            policy=mpo.policy,
            initial_state=initial_states,
            max_steps=num_simulation_steps,
        )
        trajectory = stack_list_of_tuples(trajectory)
        returns.append(trajectory.reward.sum(dim=0).mean().item())
        entropy.append(trajectory.entropy.mean())
        # Shuffle to get a state distribution
        states = trajectory.state.reshape(-1, trajectory.state.shape[-1])
        np.random.shuffle(states.numpy())
        state_batches = torch.split(states, batch_size)[::num_subsample]

    return state_batches


def _optimize_policy(mpo, state_batches, optimizer, num_gradient_steps):
    policy_episode_loss = 0.0
    value_episode_loss = 0.0
    episode_kl_div = 0.0

    # Copy over old policy for KL divergence
    mpo.reset()

    # Iterate over state batches in the state distribution
    for _ in range(num_gradient_steps):
        idx = np.random.choice(len(state_batches))
        states = state_batches[idx]
        optimizer.zero_grad()
        losses = mpo(states)
        losses.loss.backward()
        optimizer.step()

        # Track statistics
        value_episode_loss += losses.critic_loss.item()
        policy_episode_loss += losses.policy_loss.item()
        # episode_kl_div += losses.kl_div.item()
        mpo.update()

    return policy_episode_loss, value_episode_loss, episode_kl_div


def solve_mpo(
    dynamical_model,
    action_cost,
    num_iter,
    num_sim_steps,
    batch_size,
    num_gradient_steps,
    num_trajectories,
    num_action_samples,
    num_episodes,
    epsilon,
    epsilon_mean,
    epsilon_var,
    regularization,
    lr,
):
    """Solve MPO optimization problem."""
    reward_model = PendulumReward(action_cost)
    freeze_parameters(dynamical_model)
    value_function = NNValueFunction(
        dim_state=(2,),
        layers=[64, 64],
        biased_head=False,
        input_transform=StateTransform(),
    )

    policy = NNPolicy(
        dim_state=(2,),
        dim_action=(1,),
        layers=[64, 64],
        biased_head=False,
        squashed_output=True,
        input_transform=StateTransform(),
    )

    # value_function = torch.jit.script(value_function)
    init_distribution = torch.distributions.Uniform(
        torch.tensor([-np.pi, -0.05]), torch.tensor([np.pi, 0.05])
    )

    # %% Define MPC solver.
    mpo = MBMPO(
        dynamical_model,
        reward_model,
        policy,
        value_function,
        gamma=0.99,
        epsilon=epsilon,
        epsilon_mean=epsilon_mean,
        epsilon_var=epsilon_var,
        regularization=regularization,
        num_action_samples=num_action_samples,
        criterion=nn.MSELoss,
    )

    optimizer = optim.Adam([p for p in mpo.parameters() if p.requires_grad], lr=lr)

    # %% Train Controller
    test_state = torch.tensor(np.array([np.pi, 0.0]), dtype=torch.get_default_dtype())

    policy_losses, value_losses, kl_div, returns, entropy = [], [], [], [], []
    model_rewards, trajectory = 0, None

    for _ in range(num_episodes):
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.detach_test_caches():
            vloss_, ploss_, kl_div_, return_, entropy_, = train_mpo(
                mpo,
                init_distribution,
                optimizer,
                num_iter=num_iter,
                num_trajectories=num_trajectories,
                num_simulation_steps=num_sim_steps,
                num_gradient_steps=num_gradient_steps,
                batch_size=batch_size,
                num_subsample=1,
            )

        policy_losses += ploss_
        value_losses += vloss_
        returns += return_
        entropy += entropy_
        kl_div += kl_div_

        # # %% Test controller on Model.
        test_policy_on_model(
            mpo.dynamical_model, mpo.reward_model, mpo.policy, test_state
        )
        _, trajectory = test_policy_on_model(
            mpo.dynamical_model,
            mpo.reward_model,
            lambda x: (
                mpo.policy(x)[0][: mpo.dynamical_model.dim_action],
                torch.zeros(1),
            ),
            test_state,
            policy_str="Expected Policy",
        )

        model_rewards, _ = test_policy_on_model(
            mpo.dynamical_model, mpo.reward_model, mpo.policy, test_state
        )

        # %% Test controller on Environment.
        environment = SystemEnvironment(
            # ModelSystem(PendulumModel(mass=0.3, length=0.5, friction=0.005)),
            InvertedPendulum(mass=0.3, length=0.5, friction=0.005, step_size=1 / 80),
            reward=reward_model,
        )
        environment_rewards, trajectory = test_policy_on_environment(
            environment, mpo.policy, test_state
        )

        environment_rewards, _ = test_policy_on_environment(
            environment,
            lambda x: (
                mpo.policy(x)[0][: mpo.dynamical_model.dim_action],
                torch.zeros(1),
            ),
            test_state,
            policy_str="Expected Policy",
        )

    # %% Plots
    # Plot value funciton and policy.
    plot_values_and_policy(
        value_function,
        policy,
        trajectory=trajectory,
        num_entries=[200, 200],
        bounds=[(-2 * np.pi, 2 * np.pi), (-12, 12)],
    )

    # Plot returns and losses.
    plot_returns_entropy_kl(returns, entropy, kl_div)

    # Plot losses.
    plot_learning_losses(policy_losses, value_losses, horizon=20)

    return model_rewards


def get_agent_and_environment(params, agent_name):
    """Get experiment agent and environment."""
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    torch.set_num_threads(params.num_threads)

    # %% Define Environment.
    initial_distribution = torch.distributions.Uniform(
        torch.tensor([np.pi, -0.0]), torch.tensor([np.pi, +0.0])
    )
    reward_model = PendulumReward(action_cost=params.action_cost)
    environment = SystemEnvironment(
        InvertedPendulum(mass=0.3, length=0.5, friction=0.005, step_size=1 / 80),
        reward=reward_model,
        initial_state=initial_distribution.sample,
        termination_model=large_state_termination,
    )

    action_scale = environment.action_scale

    # %% Define Helper modules
    transformations = [
        ActionScaler(scale=action_scale),
        MeanFunction(DeltaState()),  # AngleWrapper(indexes=[1])
    ]

    input_transform = StateTransform()
    exploratory_distribution = torch.distributions.Uniform(
        torch.tensor([-np.pi, -0.005]), torch.tensor([np.pi, +0.005])
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
            termination_model=large_state_termination,
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
            termination_model=large_state_termination,
            initial_distribution=exploratory_distribution,
        )

    else:
        raise NotImplementedError

    # %% Add Hallucination Wrapper.
    if params.exploration == "optimistic":
        try:
            environment.add_wrapper(HallucinationWrapper)
        except AttributeError:
            environment = HallucinationWrapper(environment)
    return environment, agent


def get_mbmpo_parser():
    """Get MB-MPO parser."""
    parser = argparse.ArgumentParser(description="Parameters for Model-Based MPO.")

    parser.add_argument(
        "--exploration",
        type=str,
        default="optimistic",
        choices=["optimistic", "expected", "thompson"],
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="initial random seed (default: 0)."
    )
    parser.add_argument("--train-episodes", type=int, default=10)
    parser.add_argument("--test-episodes", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-memory", type=int, default=10000)

    environment_parser = parser.add_argument_group("environment")
    environment_parser.add_argument("--action-cost", type=float, default=0.2)
    environment_parser.add_argument("--gamma", type=float, default=0.99)
    environment_parser.add_argument("--environment-max-steps", type=int, default=400)

    model_parser = parser.add_argument_group("model")
    model_parser.add_argument(
        "--model-kind",
        type=str,
        default="ExactGP",
        choices=[
            "ExactGP",
            "SparseGP",
            "FeatureGP",
            "ProbabilisticNN",
            "DeterministicNN",
            "ProbabilisticEnsemble",
            "DeterministicEnsemble",
        ],
    )
    model_parser.add_argument("--model-learn-num-iter", type=int, default=0)
    model_parser.add_argument("--model-learn-batch-size", type=int, default=32)
    model_parser.add_argument("--not-bootstrap", action="store_true")

    model_parser.add_argument(
        "--model-sparse-approximation",
        type=str,
        default="DTC",
        choices=["DTC", "SOR", "FITC"],
    )
    model_parser.add_argument(
        "--model-feature-approximation",
        type=str,
        default="QFF",
        choices=["QFF", "RFF", "OFF"],
    )
    model_parser.add_argument("--model-opt-lr", type=float, default=5e-4)
    model_parser.add_argument("--model-opt-weight-decay", type=float, default=0)
    model_parser.add_argument("--model-max-num-points", type=int, default=int(1e10))
    model_parser.add_argument("--model-sparse-q-bar", type=int, default=2)
    model_parser.add_argument("--model-num-features", type=int, default=625)
    model_parser.add_argument("--model-layers", type=int, nargs="*", default=[64, 64])
    model_parser.add_argument("--model-non-linearity", type=str, default="ReLU")
    model_parser.add_argument("--model-unbiased-head", action="store_false")
    model_parser.add_argument("--model-num-heads", type=int, default=5)

    policy_parser = parser.add_argument_group("policy")
    policy_parser.add_argument("--policy-layers", type=int, nargs="*", default=[64, 64])
    policy_parser.add_argument("--policy-non-linearity", type=str, default="ReLU")
    policy_parser.add_argument("--policy-unbiased-head", action="store_false")
    policy_parser.add_argument("--policy-deterministic", action="store_true")
    policy_parser.add_argument("--policy-tau", type=float, default=0)

    planning_parser = parser.add_argument_group("planning")
    planning_parser.add_argument("--plan-horizon", type=int, default=1)
    planning_parser.add_argument("--plan-samples", type=int, default=8)
    planning_parser.add_argument("--plan-elites", type=int, default=1)

    value_function_parser = parser.add_argument_group("value function")
    value_function_parser.add_argument(
        "--value-function-layers", type=int, nargs="*", default=[64, 64]
    )
    value_function_parser.add_argument(
        "--value-function-non-linearity", type=str, default="ReLU"
    )
    value_function_parser.add_argument(
        "--value-function-unbiased-head", action="store_false"
    )
    policy_parser.add_argument("--value-function-tau", type=float, default=0)

    mpo_parser = parser.add_argument_group("mpo")
    mpo_parser.add_argument("--mpo-num-iter", type=int, default=50)
    mpo_parser.add_argument("--mpo-gradient-steps", type=int, default=50)
    mpo_parser.add_argument("--mpo-target-update-frequency", type=int, default=4)
    mpo_parser.add_argument("--mpo-batch_size", type=int, default=32)
    mpo_parser.add_argument("--mpo-opt-lr", type=float, default=5e-4)
    mpo_parser.add_argument("--mpo-opt-weight-decay", type=float, default=0)
    mpo_parser.add_argument("--mpo-epsilon", type=float, default=1.0)
    mpo_parser.add_argument("--mpo-epsilon-mean", type=float, default=1.7)
    mpo_parser.add_argument("--mpo-epsilon-var", type=float, default=1.1)
    mpo_parser.add_argument("--mpo-regularization", action="store_true", default=True)

    mpo_parser.add_argument("--mpo-num-action-samples", type=int, default=16)

    sim_parser = parser.add_argument_group("simulation")
    sim_parser.add_argument("--sim-num-steps", type=int, default=400)
    sim_parser.add_argument("--sim-refresh-interval", type=int, default=400)
    sim_parser.add_argument(
        "--sim-initial-states-num-trajectories", type=int, default=4
    )
    sim_parser.add_argument("--sim-initial-dist-num-trajectories", type=int, default=4)
    sim_parser.add_argument("--sim_memory-num-trajectories", type=int, default=0)
    sim_parser.add_argument("--sim_max-memory", type=int, default=100000)
    sim_parser.add_argument("--sim-num-subsample", type=int, default=1)

    viz_parser = parser.add_argument_group("visualization")
    viz_parser.add_argument("--plot-train-results", action="store_true")
    viz_parser.add_argument("--render-train", action="store_true")
    viz_parser.add_argument("--render-test", action="store_true")
    viz_parser.add_argument("--print-frequency", type=int, default=1)

    return parser
