import os
from typing import Tuple

import numpy as np
from gym import utils

try:
    from gym.envs.mujoco import mujoco_env
    class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
        action_cost: float
        prev_qpos: np.ndarray
        def __init__(self, action_cost: float = ...) -> None: ...
        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]: ...
        def _get_obs(self) -> np.ndarray: ...
        def reset_model(self) -> np.ndarray: ...
        def viewer_setup(self) -> None: ...
    class HalfCheetahEnvV2(HalfCheetahEnv):
        def __init__(self, action_cost: float = ...) -> None: ...

except Exception:  # Mujoco not installed.
    pass
