from typing import Tuple

import numpy as np
from gym import utils

try:
    from gym.envs.mujoco import mujoco_env
    class Reacher3DEnv(mujoco_env.MujocoEnv, utils.EzPickle):
        action_cost: float
        goal: np.ndarray

        sparse: bool
        length_scale: float
        action_scale: float
        def __init__(self, action_cost: float = ..., sparse: bool = ...) -> None: ...
        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]: ...
        def reset_model(self) -> np.ndarray: ...
        def _get_obs(self) -> np.ndarray: ...
        def _get_end_effector_pos(self, x: np.ndarray) -> np.ndarray: ...
        def viewer_setup(self) -> None: ...
        @staticmethod
        def get_end_effector_pos(observation: np.ndarray) -> np.ndarray: ...

except Exception:  # Mujoco not installed.
    pass
