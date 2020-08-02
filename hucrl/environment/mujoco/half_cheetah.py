"""Half Cheetah CartPole environment from https://github.com/kchua/handful-of-trials."""

import os

import numpy as np
from gym import utils

try:
    from gym.envs.mujoco import mujoco_env

    class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
        """Half Cheetah environment for MBRL control.

        References
        ----------
        Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018).
        Deep reinforcement learning in a handful of trials using probabilistic dynamics
        models. NeuRIPS.

        https://github.com/kchua/handful-of-trials
        """

        def __init__(self, action_cost=0.1):
            self.action_cost = action_cost
            self.prev_qpos = np.array([0])
            dir_path = os.path.dirname(os.path.realpath(__file__))
            mujoco_env.MujocoEnv.__init__(
                self, f"{dir_path}/assets/half_cheetah.xml", 5
            )
            utils.EzPickle.__init__(self)
            self.reset_model()

        def step(self, action: np.ndarray):
            """See `AbstractEnvironment.step()'."""
            self.prev_qpos = np.copy(self.sim.data.qpos.flat)
            self.do_simulation(action, self.frame_skip)
            ob = self._get_obs()

            reward_ctrl = -np.square(action).sum()
            reward_run = (ob[0] - self.prev_qpos[0]) / self.dt
            reward = reward_run + self.action_cost * reward_ctrl

            done = False
            return (
                ob,
                reward,
                done,
                dict(reward_run=reward_run, reward_ctrl=self.action_cost * reward_ctrl),
            )

        def _get_obs(self):
            return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])

        def reset_model(self):
            """Reset the model."""
            qpos = self.init_qpos + np.random.normal(
                loc=0, scale=0.001, size=self.model.nq
            )
            qvel = self.init_qvel + np.random.normal(
                loc=0, scale=0.001, size=self.model.nv
            )
            self.set_state(qpos, qvel)
            self.prev_qpos = np.copy(self.sim.data.qpos.flat)
            return self._get_obs()

        def viewer_setup(self):
            """Set-up the viewer."""
            self.viewer.cam.distance = self.model.stat.extent * 0.25
            self.viewer.cam.elevation = -55

    class HalfCheetahEnvV2(HalfCheetahEnv):
        """Half Cheetah V2 environment for MBRL control.

        References
        ----------
        Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018).
        Deep reinforcement learning in a handful of trials using probabilistic dynamics
        models. NeuRIPS.

        https://github.com/kchua/handful-of-trials
        """

        def __init__(self, action_cost=0.1):
            super().__init__(action_cost)

        def step(self, action: np.ndarray):
            """See `AbstractEnvironment.step()'."""
            self.prev_qpos = np.copy(self.sim.data.qpos.flat)
            self.do_simulation(action, self.frame_skip)
            ob = self._get_obs()

            reward_ctrl = -np.square(action).sum()
            reward_run = ob[0]
            reward = reward_run + self.action_cost * reward_ctrl

            done = False
            return (
                ob,
                reward,
                done,
                dict(reward_run=reward_run, reward_ctrl=self.action_cost * reward_ctrl),
            )

        def _get_obs(self):
            return np.concatenate(
                [
                    (self.sim.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
                    self.sim.data.qpos.flat[1:],
                    self.sim.data.qvel.flat,
                ]
            )

        def reset_model(self):
            """Reset the model."""
            qpos = self.init_qpos + np.random.normal(
                loc=0, scale=0.001, size=self.model.nq
            )
            qvel = self.init_qvel + np.random.normal(
                loc=0, scale=0.001, size=self.model.nv
            )
            self.set_state(qpos, qvel)
            self.prev_qpos = np.copy(self.sim.data.qpos.flat)
            return self._get_obs()


except Exception:  # Mujoco not installed.
    pass
