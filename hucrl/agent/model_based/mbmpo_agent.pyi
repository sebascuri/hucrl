"""Model-Based MPO Agent."""
from typing import Optional, Type, Union

from rllib.algorithms.mpo import MBMPO
from rllib.dataset.datatypes import Termination
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.reward import AbstractReward
from rllib.util.parameter_decay import ParameterDecay
from rllib.value_function import AbstractValueFunction
from torch.distributions import Distribution
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer

from .model_based_agent import ModelBasedAgent

class MBMPOAgent(ModelBasedAgent):
    algorithm: MBMPO
    def __init__(
        self,
        model_optimizer: Optional[Optimizer],
        policy: AbstractPolicy,
        value_function: AbstractValueFunction,
        dynamical_model: AbstractModel,
        reward_model: AbstractReward,
        optimizer: Optimizer,
        termination: Optional[Termination] = ...,
        initial_distribution: Optional[Distribution] = ...,
        plan_horizon: int = ...,
        plan_samples: int = ...,
        plan_elites: int = ...,
        max_memory: int = ...,
        model_learn_batch_size: int = ...,
        model_learn_num_iter: int = ...,
        bootstrap: bool = ...,
        mpo_value_learning_criterion: Type[_Loss] = ...,
        mpo_epsilon: Union[ParameterDecay, float] = ...,
        mpo_epsilon_mean: Union[ParameterDecay, float] = ...,
        mpo_epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        mpo_regularization: bool = ...,
        mpo_num_action_samples: int = ...,
        mpo_num_iter: int = ...,
        mpo_gradient_steps: int = ...,
        mpo_batch_size: Optional[int] = ...,
        mpo_target_update_frequency: int = ...,
        mpo_policy_update_frequency: int = ...,
        sim_num_steps: int = ...,
        sim_initial_states_num_trajectories: int = ...,
        sim_initial_dist_num_trajectories: int = ...,
        sim_memory_num_trajectories: int = ...,
        sim_num_subsample: int = ...,
        sim_max_memory: int = ...,
        sim_refresh_interval: int = ...,
        thompson_sampling: bool = ...,
        train_frequency: int = ...,
        num_rollouts: int = ...,
        gamma: float = ...,
        exploration_steps: int = ...,
        exploration_episodes: int = ...,
        tensorboard: bool = ...,
        comment: str = ...,
    ) -> None: ...
