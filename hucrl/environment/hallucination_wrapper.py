"""Hallucination Wrapper."""
from typing import Tuple

import numpy as np
from gym import Env, Wrapper
from gym.spaces import Box


class HallucinationWrapper(Wrapper):
    """A hallucination environment wrapper."""

    def __init__(self, env: Env, hallucinate_rewards: bool = False) -> None:
        env.reward_range = float("-inf"), float("+inf")
        super().__init__(env=env)
        if hallucinate_rewards:
            self.hall_shape = (self.env.observation_space.shape[0] + 1,)
        else:
            self.hall_shape = (self.env.observation_space.shape[0],)
        self.action_space = Box(
            low=np.concatenate((self.env.action_space.low, -np.ones(self.hall_shape))),
            high=np.concatenate((self.env.action_space.high, np.ones(self.hall_shape))),
            shape=(self.original_dim_action[0] + self.hall_shape[0],),
            dtype=np.float32,
        )

    @property
    def original_dim_action(self) -> Tuple[int]:
        """Get original action dimension."""
        return self.env.action_space.shape

    @property
    def hallucinated_dim_action(self) -> Tuple[int]:
        """Get original action dimension."""
        return self.hall_shape

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """See `gym.Env.step()'."""
        return self.env.step(action[: self.original_dim_action[0]])

    @property
    def name(self):
        """Get wrapper name."""
        return self.env.name
