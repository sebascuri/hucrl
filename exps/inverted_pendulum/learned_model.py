"""Solve MPO with the learned models."""

import os
import shutil

import numpy as np
import pandas as pd
import torch.distributions
from dotmap import DotMap
from rllib.model.transformed_model import TransformedModel

from exps.inverted_pendulum import (
    ACTION_COST,
    ENVIRONMENT_MAX_STEPS,
    TRAIN_EPISODES,
    get_agent_and_environment,
)
from exps.inverted_pendulum.util import solve_mpo
from exps.mb_mpo_arguments import parser

torch.manual_seed(0)
np.random.seed(0)

PLAN_HORIZON, SIM_TRAJECTORIES = 1, 8

parser.description = "Run Swing-up Inverted Pendulum using Model-Based MPO."
parser.set_defaults(
    exploration="optimistic",
    action_cost=ACTION_COST,
    train_episodes=TRAIN_EPISODES,
    environment_max_steps=ENVIRONMENT_MAX_STEPS,
    plan_horizon=PLAN_HORIZON,
    sim_num_steps=ENVIRONMENT_MAX_STEPS,
    sim_initial_states_num_trajectories=SIM_TRAJECTORIES // 2,
    sim_initial_dist_num_trajectories=SIM_TRAJECTORIES // 2,
    model_kind="ProbabilisticEnsemble",
    model_learn_num_iter=50,
    model_opt_lr=1e-3,
)

args = parser.parse_args()
params = DotMap(vars(args))

_, agent = get_agent_and_environment(params, "mbmpo")

path = "runs/Invertedpendulum/MBMPOAgent"
i = 0
df = pd.DataFrame(
    [],
    columns=[
        "exploration",
        "action_cost",
        "head",
        "model_return",
        "sim_return",
        "environment_return",
    ],
)

shutil.rmtree(agent.logger.log_dir)

for run in os.listdir(path):
    if "0.2" in run:
        action_cost = 0.2
    elif "0.1" in run:
        action_cost = 0.1
    else:
        action_cost = 0

    if params.exploration.capitalize() in run:
        try:
            statistics = pd.read_json(f"{path}/{run}/statistics.json")
        except ValueError:
            continue
        for time in [0, 4, 9]:
            agent.load(f"{path}/{run}/MBMPOAgent_{time}.pkl")
            agent_model = agent.dynamical_model
            dynamical_model = TransformedModel(
                base_model=agent_model.base_model,
                transformations=agent.dataset.transformations,
            )
            epsilon, epsilon_mean, epsilon_var = None, None, None
            eta = agent.algorithm.mpo_loss.eta().item()
            eta_mean = agent.algorithm.mpo_loss.eta_mean().item()
            eta_var = agent.algorithm.mpo_loss.eta_var().item()

            for head in range(dynamical_model.base_model.num_heads):
                dynamical_model.base_model.nn.deterministic = True
                dynamical_model.set_prediction_strategy("set_head")
                dynamical_model.base_model.set_head(head)
                model_rewards = solve_mpo(
                    dynamical_model,
                    action_cost=action_cost,
                    num_iter=4 * agent.policy_opt_num_iter,
                    num_sim_steps=agent.sim_num_steps,
                    batch_size=agent.policy_opt_batch_size,
                    num_gradient_steps=agent.policy_opt_gradient_steps,
                    num_trajectories=(
                        agent.sim_initial_states_num_trajectories
                        + agent.sim_initial_dist_num_trajectories
                    ),
                    num_action_samples=agent.algorithm.num_action_samples,
                    num_episodes=1,
                    epsilon=epsilon,
                    epsilon_mean=epsilon_mean,
                    epsilon_var=epsilon_var,
                    regularization=True,
                    lr=agent.optimizer.defaults["lr"],
                )

                df = df.append(
                    {
                        "exploration": params.exploration,
                        "action_cost": action_cost,
                        "head": head,
                        "model_return": model_rewards,
                        "sim_return": statistics.iloc[time].sim_return,
                        "environment_return": statistics.iloc[time].environment_return,
                    },
                    ignore_index=True,
                )

df.to_pickle(f"learned_{params.exploration}_DE.pkl")
