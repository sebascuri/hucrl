"""Model-Based MPO Agent."""
from itertools import chain

import torch
import torch.nn.modules.loss as loss
from rllib.model import EnsembleModel, TransformedModel
from rllib.policy import NNPolicy
from rllib.reward.quadratic_reward import QuadraticReward
from rllib.value_function import NNValueFunction
from torch.optim import Adam

from hucrl.algorithms.mbmpo import MBMPO

from .model_based_agent import ModelBasedAgent


class MBMPOAgent(ModelBasedAgent):
    """Implementation of Model-Based MPO Agent."""

    def __init__(
        self,
        model_optimizer,
        policy,
        value_function,
        dynamical_model,
        reward_model,
        optimizer,
        mpo_value_learning_criterion,
        termination_model=None,
        initial_distribution=None,
        plan_horizon=1,
        plan_samples=8,
        plan_elites=1,
        max_memory=10000,
        model_learn_batch_size=64,
        model_learn_num_iter=30,
        bootstrap=True,
        mpo_epsilon=0.1,
        mpo_epsilon_mean=0.1,
        mpo_epsilon_var=0.0001,
        mpo_regularization=False,
        mpo_num_iter=100,
        mpo_gradient_steps=50,
        mpo_batch_size=None,
        mpo_num_action_samples=15,
        mpo_target_update_frequency=4,
        mpo_policy_update_frequency=1,
        sim_num_steps=200,
        sim_initial_states_num_trajectories=8,
        sim_initial_dist_num_trajectories=0,
        sim_memory_num_trajectories=0,
        sim_max_memory=100000,
        sim_num_subsample=1,
        sim_refresh_interval=1,
        thompson_sampling=False,
        gamma=1.0,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        comment="",
    ):

        self.algorithm = MBMPO(
            dynamical_model,
            reward_model,
            policy,
            value_function,
            criterion=mpo_value_learning_criterion,
            epsilon=mpo_epsilon,
            epsilon_mean=mpo_epsilon_mean,
            epsilon_var=mpo_epsilon_var,
            regularization=mpo_regularization,
            num_action_samples=mpo_num_action_samples,
            gamma=gamma,
            termination_model=termination_model,
        )
        optimizer = type(optimizer)(
            [
                p
                for name, p in self.algorithm.named_parameters()
                if ("model" not in name and "target" not in name)
            ],
            **optimizer.defaults,
        )

        super().__init__(
            policy=policy,
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            model_optimizer=model_optimizer,
            termination_model=termination_model,
            value_function=self.algorithm.value_function,
            plan_horizon=plan_horizon,
            plan_samples=plan_samples,
            plan_elites=plan_elites,
            model_learn_num_iter=model_learn_num_iter,
            model_learn_batch_size=model_learn_batch_size,
            bootstrap=bootstrap,
            max_memory=max_memory,
            policy_opt_num_iter=mpo_num_iter,
            policy_opt_batch_size=mpo_batch_size,
            policy_opt_gradient_steps=mpo_gradient_steps,
            policy_opt_target_update_frequency=mpo_target_update_frequency,
            policy_update_frequency=mpo_policy_update_frequency,
            optimizer=optimizer,
            sim_num_steps=sim_num_steps,
            sim_initial_states_num_trajectories=sim_initial_states_num_trajectories,
            sim_initial_dist_num_trajectories=sim_initial_dist_num_trajectories,
            sim_memory_num_trajectories=sim_memory_num_trajectories,
            sim_refresh_interval=sim_refresh_interval,
            sim_num_subsample=sim_num_subsample,
            sim_max_memory=sim_max_memory,
            initial_distribution=initial_distribution,
            thompson_sampling=thompson_sampling,
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard,
            comment=comment,
        )

    @classmethod
    def default(
        cls,
        environment,
        gamma=0.99,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        test=False,
    ):
        """See `AbstractAgent.default'."""
        model = EnsembleModel(
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            num_heads=5,
            layers=[200, 200],
            biased_head=False,
            non_linearity="ReLU",
            input_transform=None,
            deterministic=False,
        )
        dynamical_model = TransformedModel(model, list())
        model_optimizer = Adam(dynamical_model.parameters(), lr=5e-4)

        reward_model = QuadraticReward(
            torch.eye(environment.dim_state[0]),
            torch.eye(environment.dim_action[0]),
            goal=environment.goal,
        )

        policy = NNPolicy(
            dim_state=environment.dim_state,
            dim_action=environment.dim_action,
            layers=[100, 100],
            biased_head=True,
            non_linearity="ReLU",
            squashed_output=True,
            input_transform=None,
            action_scale=environment.action_scale,
            goal=environment.goal,
            deterministic=False,
            tau=5e-3,
        )

        value_function = NNValueFunction(
            dim_state=environment.dim_state,
            layers=[200, 200],
            biased_head=True,
            non_linearity="ReLU",
            input_transform=None,
            tau=5e-3,
        )

        optimizer = Adam(
            chain(policy.parameters(), value_function.parameters()), lr=5e-3
        )

        return cls(
            model_optimizer,
            policy,
            value_function,
            dynamical_model,
            reward_model,
            optimizer,
            mpo_value_learning_criterion=loss.MSELoss,
            termination_model=None,
            initial_distribution=None,
            plan_horizon=1,
            plan_samples=8,
            plan_elites=1,
            max_memory=10000,
            model_learn_batch_size=64,
            model_learn_num_iter=4 if test else 30,
            bootstrap=True,
            mpo_epsilon=0.1,
            mpo_epsilon_mean=0.1,
            mpo_epsilon_var=0.0001,
            mpo_regularization=False,
            mpo_num_iter=5 if test else 200,
            mpo_gradient_steps=50,
            mpo_batch_size=None,
            mpo_num_action_samples=15,
            mpo_target_update_frequency=4,
            sim_num_steps=5 if test else 200,
            sim_initial_states_num_trajectories=8,
            sim_initial_dist_num_trajectories=0,
            sim_memory_num_trajectories=0,
            sim_max_memory=100000,
            sim_num_subsample=1,
            sim_refresh_interval=1,
            thompson_sampling=False,
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard,
            comment=environment.name,
        )
