"""Solve MPO with true model."""

import numpy as np
import torch.distributions

from exps.inverted_pendulum.util import PendulumModel, solve_mpo

torch.manual_seed(0)
np.random.seed(0)

dynamical_model = PendulumModel(mass=0.3, length=0.5, friction=0.005, step_size=1 / 80)


batch_size = 32
num_action_samples = 16
num_trajectories = 16
num_episodes = 1
# epsilon, epsilon_mean, epsilon_var = None, None, None
# eta, eta_mean, eta_var = 1., 1.7, 1.1
epsilon, epsilon_mean, epsilon_var = 0.1, 0.01, 0.0001
eta, eta_mean, eta_var = None, None, None
lr = 5e-4

num_iter = 200
num_sim_steps = 400
num_gradient_steps = 50

for action_cost in [0.0, 0.1, 0.2]:
    returns = solve_mpo(
        dynamical_model,
        action_cost=action_cost,
        num_iter=num_iter,
        num_sim_steps=num_sim_steps,
        batch_size=batch_size,
        num_gradient_steps=num_gradient_steps,
        num_trajectories=num_trajectories,
        num_action_samples=num_action_samples,
        num_episodes=num_episodes,
        epsilon=epsilon,
        epsilon_mean=epsilon_mean,
        epsilon_var=epsilon_var,
        regularization=False,
        lr=lr,
    )
    print(np.max(returns))
