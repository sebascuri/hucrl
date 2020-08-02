from typing import Any, NamedTuple, Optional, Tuple, Union

import torch.nn as nn
from rllib.algorithms.abstract_algorithm import AbstractAlgorithm, MPOLoss
from rllib.dataset.datatypes import Observation, Termination
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.reward import AbstractReward
from rllib.util.parameter_decay import ParameterDecay
from rllib.util.utilities import RewardTransformer
from rllib.value_function import AbstractQFunction, AbstractValueFunction
from torch import Tensor
from torch.distributions import Distribution
from torch.nn.modules.loss import _Loss

class MPOLosses(NamedTuple):
    primal_loss: Tensor
    dual_loss: Tensor

class MPOWorker(nn.Module):
    eta: ParameterDecay
    eta_mean: ParameterDecay
    eta_var: ParameterDecay

    epsilon: Tensor
    epsilon_mean: Tensor
    epsilon_var: Tensor
    def __init__(
        self,
        epsilon: Union[ParameterDecay, float] = ...,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        regularization: bool = ...,
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> MPOLosses: ...

class MPO(AbstractAlgorithm):
    old_policy: AbstractPolicy
    policy: AbstractPolicy
    q_function: AbstractQFunction
    q_target: AbstractQFunction

    gamma: float
    num_action_samples: int

    mpo_loss: MPOWorker
    value_loss: _Loss
    reward_transformer: RewardTransformer
    def __init__(
        self,
        policy: AbstractPolicy,
        q_function: AbstractQFunction,
        num_action_samples: int,
        criterion: _Loss,
        epsilon: Union[ParameterDecay, float] = ...,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        regularization: bool = ...,
        reward_transformer: RewardTransformer = ...,
        gamma: float = ...,
    ) -> None: ...
    def get_kl_and_pi(self, state: Tensor) -> Tuple[Tensor, Tensor, Distribution]: ...
    def get_value_target(
        self, reward: Tensor, next_state: Tensor, done: Tensor
    ) -> Tensor: ...
    def reset(self) -> None: ...
    def forward(self, observation: Observation, **kwargs: Any) -> MPOLoss: ...

class MBMPO(AbstractAlgorithm):
    dynamical_model: AbstractModel
    reward_model: AbstractReward
    policy: AbstractPolicy
    value_function: AbstractValueFunction
    value_function_target: AbstractValueFunction

    gamma: float

    mpo_loss: MPOWorker
    value_loss: _Loss
    num_action_samples: int
    entropy_reg: float
    reward_transformer: RewardTransformer
    termination: Optional[Termination]
    dist_params: dict
    def __init__(
        self,
        dynamical_model: AbstractModel,
        reward_model: AbstractReward,
        policy: AbstractPolicy,
        value_function: AbstractValueFunction,
        criterion: _Loss,
        epsilon: Union[ParameterDecay, float] = ...,
        epsilon_mean: Union[ParameterDecay, float] = ...,
        epsilon_var: Optional[Union[ParameterDecay, float]] = ...,
        regularization: bool = ...,
        gamma: float = ...,
        num_action_samples: int = ...,
        reward_transformer: RewardTransformer = ...,
        termination: Optional[Termination] = ...,
    ) -> None: ...
    def reset(self) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> MPOLoss: ...
