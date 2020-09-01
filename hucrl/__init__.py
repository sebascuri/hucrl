"""Python Script Template."""
import gym
import numpy as np
import torch

from hucrl.environment import *

torch.set_default_dtype(torch.float32)
gym.logger.set_level(gym.logger.ERROR)
np.set_printoptions(precision=3)
