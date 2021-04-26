from typing import Any, List, Tuple, Union, Type, TypeVar

import torch.nn as nn
from rllib.dataset.datatypes import TupleDistribution
from rllib.model.abstract_model import AbstractModel
from rllib.model.transformed_model import TransformedModel
from torch import Tensor

T = TypeVar("T", bound="HallucinatedModel")

class HallucinatedModel(TransformedModel):
    """Optimistic Model returns a Delta at the optimistic next state."""

    a_dim_action: Tuple
    h_dim_action: Tuple
    beta: float
    def __init__(
        self,
        base_model: AbstractModel,
        transformations: Union[List[nn.Module], nn.ModuleList],
        beta: float = ...,
        hallucinate_rewards: bool = ...,
    ) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
    @classmethod
    def from_transformed_model(
        cls: Type[T],
        transformed_model: TransformedModel,
        beta: float = ...,
        hallucinate_rewards: bool = ...,
    ) -> T: ...
