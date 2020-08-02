"""Learning a value function with a model."""

import numpy as np
import rllib.algorithms.control
import rllib.util.neural_networks
import torch
import torch.nn as nn
import torch.optim as optim
from rllib.environment import SystemEnvironment
from rllib.environment.systems import InvertedPendulum
from rllib.model import LinearModel
from rllib.policy import NNPolicy
from rllib.reward.quadratic_reward import QuadraticReward
from rllib.util import mb_return
from rllib.value_function import NNValueFunction
from tqdm import tqdm

from exps.inverted_pendulum.plotters import plot_learning_losses, plot_values_and_policy

num_steps = 1
discount = 1.0
batch_size = 40

non_linear_system = InvertedPendulum(mass=0.1, length=0.5, friction=0.0)
system = non_linear_system.linearize()
q = np.eye(2)
r = 0.01 * np.eye(1)
gamma = 0.99

K, P = rllib.algorithms.control.dlqr(system.a, system.b, q, r, gamma=gamma)
K = torch.from_numpy(K.T).type(torch.get_default_dtype())
P = torch.from_numpy(P).type(torch.get_default_dtype())

reward_model = QuadraticReward(
    torch.from_numpy(q).type(torch.get_default_dtype()),
    torch.from_numpy(r).type(torch.get_default_dtype()),
)
environment = SystemEnvironment(
    system, initial_state=None, termination=None, reward=reward_model
)

model = LinearModel(system.a, system.b)

policy = NNPolicy(
    dim_state=system.dim_state,
    dim_action=system.dim_action,
    layers=[],
    biased_head=False,
    deterministic=True,
)  # Linear policy.
print(f"initial: {policy.nn.head.weight}")  # type: ignore

value_function = NNValueFunction(
    dim_state=system.dim_state, layers=[64, 64], biased_head=False
)

# policy = torch.jit.script(policy)
model = torch.jit.script(model)
value_function = torch.jit.script(value_function)

loss_function = nn.MSELoss()
value_optimizer = optim.Adam(value_function.parameters(), lr=5e-4)
policy_optimizer = optim.Adam(policy.parameters(), lr=5e-3)

policy_losses = []
value_losses = []
torch.autograd.set_detect_anomaly(True)

num_iter = 10000
for i in tqdm(range(num_iter)):
    value_optimizer.zero_grad()
    policy_optimizer.zero_grad()

    states = 0.5 * torch.randn(batch_size, 2)
    with rllib.util.neural_networks.disable_gradient(value_function):
        value_estimate, trajectory = mb_return(
            state=states,
            dynamical_model=model,
            policy=policy,
            reward_model=reward_model,
            num_steps=1,
            gamma=gamma,
            value_function=value_function,
            num_samples=15,
        )

    prediction = value_function(states)
    value_loss = loss_function(prediction, value_estimate.mean(dim=0))
    policy_loss = -value_estimate.mean()

    loss = policy_loss + value_loss
    loss.backward()
    policy_optimizer.step()
    value_optimizer.step()

    policy_losses.append(policy_loss.item())
    value_losses.append(value_loss.item())

horizon = 20
plot_learning_losses(policy_losses, value_losses, horizon)

print(f"optimal: {K}")
print(f"learned: {policy.nn.head.weight}")  # type: ignore

bounds = [(-0.5, 0.5), (-0.5, 0.5)]
num_entries = [100, 100]

plot_values_and_policy(
    lambda x: rllib.util.neural_networks.torch_quadratic(x, matrix=-P),
    lambda x: (x @ K, 0),
    bounds,
    num_entries,
    suptitle="Exact",
)

plot_values_and_policy(value_function, policy, bounds, num_entries, suptitle="Learnt")
