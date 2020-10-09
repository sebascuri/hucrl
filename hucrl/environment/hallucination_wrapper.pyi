"""Base Class of an Adversarial Environments."""
from typing import Tuple

import numpy as np
from gym import Env, Wrapper
from gym.spaces import Box

class HallucinationWrapper(Wrapper):
    """A hallucination environment wrapper."""

    action_space: Box
    def __init__(self, env: Env, hallucinate_rewards: bool = ...) -> None: ...
    @property
    def original_dim_action(self) -> Tuple[int]: ...
    @property
    def hallucinated_dim_action(self) -> Tuple[int]: ...
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]: ...
