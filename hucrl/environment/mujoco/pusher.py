"""Mujoco Pusher environment from https://github.com/kchua/handful-of-trials."""

import os

import numpy as np
from gym import utils

try:
    from gym.envs.mujoco import mujoco_env

    class PusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
        """Pusher environment for MBRL control.

        References
        ----------
        Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018).
        Deep reinforcement learning in a handful of trials using probabilistic dynamics
        models. NeuRIPS.

        https://github.com/kchua/handful-of-trials
        """

        def __init__(self, action_cost=0.1):
            self.action_cost = action_cost
            dir_path = os.path.dirname(os.path.realpath(__file__))
            mujoco_env.MujocoEnv.__init__(self, f"{dir_path}/assets/pusher.xml", 4)
            utils.EzPickle.__init__(self)
            self.goal_pos = np.asarray([0, 0])
            self.cylinder_pos = np.array([-0.25, 0.15]) + np.random.normal(
                0, 0.025, [2]
            )
            # self._goal = self.get_body_com("goal")
            self.reset_model()

        def step(self, action: np.ndarray):
            """See `AbstractEnvironment.step()'."""
            obj_pos = self.get_body_com("object")  # type: np.ndarray
            dist_to_obj = obj_pos - self.get_body_com("tips_arm")  # type: np.ndarray
            dist_to_goal = obj_pos - self.get_body_com("goal")  # type: np.ndarray

            reward_dist_to_obj = -np.sum(np.abs(dist_to_obj))
            reward_dist_to_goal = -np.sum(np.abs(dist_to_goal)[:-1])
            reward_ctrl = -np.square(action).sum()
            reward_state = 1.25 * reward_dist_to_goal + 0.5 * reward_dist_to_obj
            reward = reward_state + self.action_cost * reward_ctrl

            self.do_simulation(action, self.frame_skip)
            ob = self._get_obs()
            done = False
            return (
                ob,
                reward,
                done,
                dict(
                    reward_dist_to_goal=1.25 * reward_dist_to_goal,
                    reward_dist_to_obj=0.5 * reward_dist_to_obj,
                    reward_ctrl=self.action_cost * reward_ctrl,
                ),
            )

        @staticmethod
        def get_end_effector_pos(observation):
            """Get end effector position."""
            theta1, theta2, theta3, theta4, theta5, theta6, *_ = np.split(
                observation, observation.shape[-1], -1
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
                    0.1 * np.sin(theta1) + 0.4 * np.sin(theta1) * np.cos(theta2) - 0.6,
                    -0.4 * np.sin(theta2),
                ],
                axis=1,
            )

            for length, hinge, roll in [(0.321, theta4, theta3)]:
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

        def viewer_setup(self):
            """Set-up the viewer."""
            self.viewer.cam.trackbodyid = -1
            self.viewer.cam.distance = 4.0

        def reset_model(self):
            """Reset the model."""
            qpos = self.init_qpos

            self.goal_pos = np.asarray([0, 0])
            self.cylinder_pos = np.array([-0.25, 0.15]) + np.random.normal(
                0, 0.025, [2]
            )

            qpos[-4:-2] = self.cylinder_pos
            qpos[-2:] = self.goal_pos
            qvel = self.init_qvel + self.np_random.uniform(
                low=-0.005, high=0.005, size=self.model.nv
            )
            qvel[-4:] = 0
            self.set_state(qpos, qvel)
            self._goal = self.get_body_com("goal")

            return self._get_obs()

        def _get_obs(self):
            return np.concatenate(
                [
                    self.sim.data.qpos.flat[:7],
                    self.sim.data.qvel.flat[:7],
                    self.get_body_com("object"),
                    # self.get_body_com("tips_arm"),
                ]
            )


except Exception:  # Mujoco not installed.
    pass
