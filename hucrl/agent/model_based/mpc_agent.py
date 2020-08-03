"""MPC Agent Implementation."""

import torch
from rllib.algorithms.mpc import CEMShooting
from rllib.model import EnsembleModel, TransformedModel
from rllib.policy.mpc_policy import MPCPolicy
from rllib.reward.quadratic_reward import QuadraticReward
from torch.optim import Adam

from .model_based_agent import ModelBasedAgent


class MPCAgent(ModelBasedAgent):
    """Implementation of an agent that runs an MPC policy."""

    def __init__(
        self,
        mpc_policy: MPCPolicy,
        model_learn_num_iter=0,
        model_learn_batch_size=64,
        bootstrap=True,
        model_optimizer=None,
        max_memory=1,
        sim_num_steps=0,
        sim_initial_states_num_trajectories=0,
        sim_initial_dist_num_trajectories=0,
        sim_memory_num_trajectories=0,
        initial_distribution=None,
        thompson_sampling=False,
        gamma=1.0,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        comment="",
    ):
        super().__init__(
            dynamical_model=mpc_policy.solver.dynamical_model,
            reward_model=mpc_policy.solver.reward_model,
            policy=mpc_policy,
            value_function=mpc_policy.solver.terminal_reward,
            termination=mpc_policy.solver.termination,
            model_optimizer=model_optimizer,
            plan_horizon=0,  # Calling the mpc policy already plans.
            plan_samples=0,
            plan_elites=0,
            model_learn_num_iter=model_learn_num_iter,
            model_learn_batch_size=model_learn_batch_size,
            bootstrap=bootstrap,
            policy_opt_num_iter=0,
            max_memory=max_memory,
            sim_num_steps=sim_num_steps,
            sim_initial_states_num_trajectories=sim_initial_states_num_trajectories,
            sim_initial_dist_num_trajectories=sim_initial_dist_num_trajectories,
            sim_memory_num_trajectories=sim_memory_num_trajectories,
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

        mpc_solver = CEMShooting(
            dynamical_model,
            reward_model,
            5 if test else 25,
            gamma=gamma,
            num_iter=2 if test else 5,
            num_samples=20 if test else 400,
            num_elites=5 if test else 40,
            termination=None,
            terminal_reward=None,
            warm_start=True,
            default_action="zero",
            num_cpu=1,
        )
        policy = MPCPolicy(mpc_solver)

        return cls(
            policy,
            model_learn_num_iter=4 if test else 30,
            model_learn_batch_size=64,
            bootstrap=True,
            model_optimizer=model_optimizer,
            max_memory=1,
            sim_num_steps=0,
            sim_initial_states_num_trajectories=0,
            sim_initial_dist_num_trajectories=0,
            sim_memory_num_trajectories=0,
            initial_distribution=None,
            thompson_sampling=False,
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard,
            comment=environment.name,
        )
