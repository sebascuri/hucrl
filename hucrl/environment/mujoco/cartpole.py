"""Mujoco CartPole environment from https://github.com/kchua/handful-of-trials."""
import os

import numpy as np
from gym import utils

try:
    from gym.envs.mujoco import mujoco_env

    class CartPoleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
        """CartPole environment for MBRL control.

        References
        ----------
        Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018).
        Deep reinforcement learning in a handful of trials using probabilistic dynamics
        models. NeuRIPS.

        https://github.com/kchua/handful-of-trials
        """

        def __init__(self, action_cost=0.01, pendulum_length=0.6):
            self.pendulum_length = pendulum_length
            self.action_cost = action_cost
            utils.EzPickle.__init__(self)
            dir_path = os.path.dirname(os.path.realpath(__file__))
            mujoco_env.MujocoEnv.__init__(self, f"{dir_path}/assets/cartpole.xml", 2)

        def step(self, action: np.ndarray):
            """See `AbstractEnvironment.step()'."""
            self.do_simulation(action, self.frame_skip)
            ob = self._get_obs()

            pendulum_length = self.pendulum_length
            end_effector = self._get_end_effector_pos(ob)

            reward_dist = np.exp(
                -np.sum(np.square(end_effector)) / (pendulum_length ** 2)
            )
            reward_ctrl = -np.sum(np.square(action))
            reward = reward_dist + self.action_cost * reward_ctrl

            done = False
            return (
                ob,
                reward,
                done,
                dict(
                    reward_dist=reward_dist, reward_ctrl=self.action_cost * reward_ctrl
                ),
            )

        def reset_model(self):
            """Reset the model."""
            qpos = self.init_qpos + np.random.normal(0, 0.1, np.shape(self.init_qpos))
            qvel = self.init_qvel + np.random.normal(0, 0.1, np.shape(self.init_qvel))
            self.set_state(qpos, qvel)
            return self._get_obs()

        def _get_obs(self):
            return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

        def _get_end_effector_pos(self, x):
            x0, theta = x[0], x[1]
            sin, cos = np.sin(theta), np.cos(theta)
            pendulum_length = self.pendulum_length
            return np.array([x0 - pendulum_length * sin, -pendulum_length * (cos + 1)])

        def viewer_setup(self):
            """Set-up the viewer."""
            v = self.viewer
            v.cam.trackbodyid = 0
            v.cam.distance = self.model.stat.extent


except Exception:  # Mujoco not installed.
    pass
