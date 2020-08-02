"""Mujoco Reacher environment from https://github.com/kchua/handful-of-trials."""

import os

import numpy as np
from gym import utils

try:
    from gym.envs.mujoco import mujoco_env

    class Reacher3DEnv(mujoco_env.MujocoEnv, utils.EzPickle):
        """Reacher environment for MBRL control.

        References
        ----------
        Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018).
        Deep reinforcement learning in a handful of trials using probabilistic dynamics
        models. NeuRIPS.

        https://github.com/kchua/handful-of-trials
        """

        def __init__(self, action_cost=0.01, sparse=False):
            self.action_cost = action_cost
            self.length_scale = 0.45  # .5 is solved by all.
            self.action_scale = 2.0  # 5.
            self.goal = np.zeros(3)
            self.sparse = sparse

            utils.EzPickle.__init__(self)
            dir_path = os.path.dirname(os.path.realpath(__file__))
            mujoco_env.MujocoEnv.__init__(self, "%s/assets/reacher3d.xml" % dir_path, 2)

        def step(self, action: np.ndarray):
            """See `AbstractEnvironment.step()'."""
            self.do_simulation(action, self.frame_skip)
            ob = self._get_obs()
            dist = self.get_end_effector_pos(ob) - self.goal
            if self.sparse:
                reward_dist = np.exp(
                    -np.sum(np.square(dist)) / (self.length_scale ** 2)
                )
                reward_ctrl = (
                    np.exp(-np.sum(np.square(action)) / (self.action_scale ** 2)) - 1
                )
            else:
                reward_dist = -np.sum(np.square(dist))
                reward_ctrl = -np.square(action).sum()

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

        def viewer_setup(self):
            """Set-up the viewer."""
            self.viewer.cam.trackbodyid = 1
            self.viewer.cam.distance = 2.5
            self.viewer.cam.elevation = -30
            self.viewer.cam.azimuth = 270

        def reset_model(self):
            """Reset the model."""
            qpos, qvel = np.copy(self.init_qpos), np.copy(self.init_qvel)
            qpos[-3:] += np.random.normal(loc=0, scale=0.1, size=[3])
            qvel[-3:] = 0
            self.goal = qpos[-3:]
            self.set_state(qpos, qvel)
            return self._get_obs()

        def _get_obs(self):
            return np.concatenate(
                [self.sim.data.qpos.flat[:-3], self.sim.data.qvel.flat[:-3]]
            )

        @staticmethod
        def get_end_effector_pos(states):
            """Get end effector position."""
            theta1, theta2, theta3, theta4, theta5, theta6, *_ = np.split(
                states, len(states), -1
            )
            rot_axis = np.stack(
                [
                    np.cos(theta2) * np.cos(theta1),
                    np.cos(theta2) * np.sin(theta1),
                    -np.sin(theta2),
                ],
                axis=1,
            )
            rot_perp_axis = np.stack(
                [-np.sin(theta1), np.cos(theta1), np.zeros(theta1.shape)], axis=1
            )

            cur_end = np.stack(
                [
                    0.1 * np.cos(theta1) + 0.4 * np.cos(theta1) * np.cos(theta2),
                    0.1 * np.sin(theta1)
                    + 0.4 * np.sin(theta1) * np.cos(theta2)
                    - 0.188,
                    -0.4 * np.sin(theta2),
                ],
                axis=1,
            )

            for length, hinge, roll in [
                (0.321, theta4, theta3),
                (0.16828, theta6, theta5),
            ]:
                perp_all_axis = np.cross(rot_axis, rot_perp_axis)
                x = np.cos(hinge) * rot_axis
                y = np.sin(hinge) * np.sin(roll) * rot_perp_axis
                z = -np.sin(hinge) * np.cos(roll) * perp_all_axis
                new_rot_axis = x + y + z
                new_rot_perp_axis = np.cross(new_rot_axis, rot_axis)
                new_rot_perp_axis[
                    np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30
                ] = rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30]
                new_rot_perp_axis /= np.linalg.norm(
                    new_rot_perp_axis, axis=1, keepdims=True
                )
                rot_axis, rot_perp_axis = new_rot_axis, new_rot_perp_axis
                cur_end = cur_end + length * new_rot_axis

            return cur_end


except Exception:  # Mujoco not installed.
    pass
