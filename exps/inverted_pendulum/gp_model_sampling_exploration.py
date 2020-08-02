"""Solve MPO with a GP model and collecting data querying randomly a simulator."""

from typing import List

import numpy as np
import torch
from rllib.dataset.experience_replay import BootstrapExperienceReplay
from rllib.dataset.transforms import (
    AbstractTransform,
    ActionClipper,
    DeltaState,
    MeanFunction,
)
from rllib.model.abstract_model import AbstractModel
from rllib.model.gp_model import ExactGPModel
from rllib.model.transformed_model import TransformedModel
from rllib.util.collect_data import collect_model_transitions
from rllib.util.training import train_model
from torch.distributions import Uniform
from torch.utils.data import DataLoader

from exps.inverted_pendulum.util import (
    PendulumModel,
    PendulumReward,
    StateTransform,
    solve_mpo,
)

torch.manual_seed(0)
np.random.seed(0)

# %% Collect Data.
num_data = 200
reward_model = PendulumReward()
dynamical_model = PendulumModel(
    mass=0.3, length=0.5, friction=0.005, step_size=1 / 80
)  # type: AbstractModel

transitions = collect_model_transitions(
    Uniform(torch.tensor([-2 * np.pi, -12]), torch.tensor([2 * np.pi, 12])),
    Uniform(torch.tensor([-1.0]), torch.tensor([1.0])),
    dynamical_model,
    reward_model,
    num_data,
)

# %% Bootstrap into different trajectories.
transformations = [
    ActionClipper(-1, 1),
    MeanFunction(DeltaState()),
    # StateActionNormalizer()
]  # type: List[AbstractTransform]
dataset = BootstrapExperienceReplay(
    max_len=int(1e4), transformations=transformations, num_bootstraps=1
)
for transition in transitions:
    dataset.append(transition)

data = dataset.all_data
split = 50
# dataset._ptr = split
train_loader = DataLoader(dataset, batch_size=split, shuffle=False)

# %% Train a Model
model = ExactGPModel(
    data.state[:split, 0],
    data.action[:split, 0],
    data.next_state[:split, 0],
    input_transform=StateTransform(),
    max_num_points=75,
)

model.eval()
mean, stddev = model(torch.randn(8, 5, 2), torch.randn(8, 5, 1))

optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)
print([(gp.output_scale, gp.length_scale) for gp in model.gp])
train_model(model, train_loader, optimizer, max_iter=100)
print([(gp.output_scale, gp.length_scale) for gp in model.gp])

# %% Add data and re-train
model.add_data(
    data.state[split : 2 * split, 0],
    data.action[split : 2 * split, 0],
    data.next_state[split : 2 * split, 0],
)

train_model(model, train_loader, optimizer, max_iter=70)
print([(gp.output_scale, gp.length_scale) for gp in model.gp])

model.add_data(
    data.state[2 * split :, 0],
    data.action[2 * split :, 0],
    data.next_state[2 * split :, 0],
)

# %% Define dynamical model.
model.gp[0].output_scale = torch.tensor(1.0)
model.gp[0].length_scale = torch.tensor([[9.0]])
model.likelihood[0].noise = torch.tensor([1e-4])

model.gp[1].output_scale = torch.tensor(1.0)
model.gp[1].length_scale = torch.tensor([[9.0]])
model.likelihood[1].noise = torch.tensor([1e-4])

for gp in model.gp:
    gp.prediction_strategy = None

model.eval()
dynamical_model = TransformedModel(model, transformations)

# %% SOLVE MPC
action_cost = 0.2

batch_size = 32
num_action_samples = 16
num_trajectories = 8
num_episodes = 1
epsilon, epsilon_mean, epsilon_var = None, None, None
eta, eta_mean, eta_var = 1.0, 1.7, 1.1
lr = 5e-4

num_iter = 100
num_sim_steps = 400
num_gradient_steps = 50

solve_mpo(
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
    regularization=True,
    lr=lr,
)
