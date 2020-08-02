from typing import Any, Tuple

from rllib.dataset.datatypes import TupleDistribution
from rllib.policy.abstract_policy import AbstractPolicy
from torch import Tensor

class DerivedPolicy(AbstractPolicy):

    base_policy: AbstractPolicy
    def __init__(self, base_policy: AbstractPolicy, dim_action: Tuple) -> None: ...
    def forward(self, *args: Tensor, **kwargs: Any) -> TupleDistribution: ...
