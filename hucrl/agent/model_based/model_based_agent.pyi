from typing import Optional

from rllib.agent.abstract_agent import AbstractAgent
from rllib.algorithms.abstract_algorithm import AbstractAlgorithm
from rllib.dataset.datatypes import Observation
from rllib.dataset.experience_replay import ExperienceReplay, StateExperienceReplay
from rllib.model import AbstractModel
from rllib.model.transformed_model import TransformedModel
from rllib.policy import AbstractPolicy
from rllib.value_function import AbstractValueFunction
from torch import Tensor
from torch.distributions import Distribution
from torch.optim.optimizer import Optimizer

class ModelBasedAgent(AbstractAgent):
    dynamical_model: TransformedModel
    reward_model: AbstractModel
    termination_model: Optional[AbstractModel]
    value_function: AbstractValueFunction

    model_optimizer: Optimizer
    dataset: ExperienceReplay
    sim_dataset: StateExperienceReplay
    sim_trajectory: Optional[Observation]

    model_learn_num_iter: int
    model_learn_batch_size: int
    plan_horizon: int
    plan_samples: int
    plan_elites: int

    algorithm: AbstractAlgorithm
    policy: AbstractPolicy
    policy_opt_num_iter: int
    policy_opt_batch_size: int
    policy_opt_gradient_steps: int
    policy_opt_target_update_frequency: int
    optimizer: Optional[Optimizer]

    sim_num_steps: int
    sim_initial_states_num_trajectories: int
    sim_initial_dist_num_trajectories: int
    sim_memory_num_trajectories: int
    sim_refresh_interval: int
    sim_num_subsample: int
    initial_distribution: Distribution
    initial_states: StateExperienceReplay
    new_episode: bool
    thompson_sampling: bool
    def __init__(
        self,
        dynamical_model: AbstractModel,
        reward_model: AbstractModel,
        policy: AbstractPolicy,
        model_optimizer: Optional[Optimizer] = ...,
        value_function: Optional[AbstractValueFunction] = ...,
        termination_model: Optional[AbstractModel] = ...,
        plan_horizon: int = ...,
        plan_samples: int = ...,
        plan_elites: int = ...,
        model_learn_num_iter: int = ...,
        model_learn_batch_size: int = ...,
        bootstrap: bool = ...,
        max_memory: int = ...,
        policy_opt_num_iter: int = ...,
        policy_opt_batch_size: Optional[int] = ...,
        policy_opt_gradient_steps: int = ...,
        policy_opt_target_update_frequency: int = ...,
        policy_update_frequency: int = ...,
        optimizer: Optional[Optimizer] = ...,
        sim_num_steps: int = ...,
        sim_initial_states_num_trajectories: int = ...,
        sim_initial_dist_num_trajectories: int = ...,
        sim_memory_num_trajectories: int = ...,
        sim_refresh_interval: int = ...,
        sim_num_subsample: int = ...,
        sim_max_memory: int = ...,
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
    def plan(self, state: Tensor) -> Tensor: ...
    def learn(self) -> None: ...
    def learn_model(self) -> None: ...
    def _log_simulated_trajectory(self) -> None: ...
    def simulate_and_learn_policy(self): ...
    def simulate_model(self): ...
    def learn_policy(self) -> None: ...
