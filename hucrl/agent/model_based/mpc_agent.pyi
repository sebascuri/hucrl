"""MPC Agent Implementation."""

from typing import Optional

from rllib.policy.mpc_policy import MPCPolicy
from torch.distributions import Distribution
from torch.optim.optimizer import Optimizer

from .model_based_agent import ModelBasedAgent

class MPCAgent(ModelBasedAgent):
    """Implementation of an agent that runs an MPC policy."""

    def __init__(
        self,
        mpc_policy: MPCPolicy,
        model_learn_num_iter: int = ...,
        model_learn_batch_size: int = ...,
        bootstrap: bool = ...,
        model_optimizer: Optional[Optimizer] = ...,
        max_memory: int = ...,
        sim_num_steps: int = ...,
        sim_initial_states_num_trajectories: int = ...,
        sim_initial_dist_num_trajectories: int = ...,
        sim_memory_num_trajectories: int = ...,
        initial_distribution: Optional[Distribution] = ...,
        thompson_sampling: bool = ...,
        train_frequency: int = ...,
        num_rollouts: int = ...,
        gamma: float = ...,
        exploration_steps: int = ...,
        exploration_episodes: int = ...,
        tensorboard: bool = ...,
        comment: str = ...,
    ) -> None: ...
