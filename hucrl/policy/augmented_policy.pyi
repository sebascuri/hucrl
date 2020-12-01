from typing import Any

from rllib.policy import AbstractPolicy

class AugmentedPolicy(AbstractPolicy):
    true_policy: AbstractPolicy
    hallucination_policy: AbstractPolicy
    def __init__(
        self,
        true_policy: AbstractPolicy,
        hallucination_policy: AbstractPolicy,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
