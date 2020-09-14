from typing import Any, Optional, Union

import torch.nn as nn
from rllib.algorithms.abstract_algorithm import AbstractAlgorithm, Loss
from rllib.algorithms.kl_loss import KLLoss
from rllib.algorithms.mpo import MPOLoss
from rllib.model import AbstractModel
from rllib.policy import AbstractPolicy
from rllib.util.parameter_decay import ParameterDecay
from rllib.util.utilities import RewardTransformer
from rllib.value_function import AbstractValueFunction
from torch import Tensor
from torch.nn.modules.loss import _Loss

class MBMPO(AbstractAlgorithm):
    dynamical_model: AbstractModel
    reward_model: AbstractModel
    policy: AbstractPolicy
    value_function: AbstractValueFunction
    value_function_target: AbstractValueFunction

    mpo_loss: MPOLoss
    kl_loss: KLLoss
    value_loss: _Loss
    num_action_samples: int
    entropy_reg: float
    reward_transformer: RewardTransformer
    termination_model: Optional[AbstractModel]
    dist_params: dict
    def __init__(
        self,
        dynamical_model: AbstractModel,
        reward_model: AbstractModel,
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
        termination_model: Optional[AbstractModel] = ...,
    ) -> None: ...
    def reset(self) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> Loss: ...
