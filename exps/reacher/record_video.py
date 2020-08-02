"""Record video."""

import shutil

from dotmap import DotMap
from rllib.util.rollout import record

from exps.mpc_arguments import parser
from exps.reacher import (
    ACTION_COST,
    ENVIRONMENT_MAX_STEPS,
    TRAIN_EPISODES,
    get_agent_and_environment,
)

MPC_HORIZON, MPC_NUM_SAMPLES = 25, 400

parser.description = "Run Reacher using Model-Based MPC."
parser.set_defaults(
    action_cost=ACTION_COST,
    train_episodes=TRAIN_EPISODES,
    environment_max_steps=ENVIRONMENT_MAX_STEPS,
    mpc_horizon=MPC_HORIZON,
    mpc_num_samples=MPC_NUM_SAMPLES,
    mpc_num_elites=MPC_NUM_SAMPLES // 10,
    model_kind="ProbabilisticEnsemble",
    model_learn_num_iter=50,
    max_memory=10 * ENVIRONMENT_MAX_STEPS,
    model_layers=[256, 256, 256],
    model_non_linearity="swish",
    model_opt_lr=1e-4,
    model_opt_weight_decay=0.0005,
)

args = parser.parse_args()
params = DotMap(vars(args))

environment, agent = get_agent_and_environment(params, "mpc")
shutil.rmtree(agent.logger.writer.logdir)

for penalty in [0.1, 0.01, 0.05]:
    path = "runs/MPCAgent/"
    agent.load(f"{path}/{penalty}/MPCAgent_best.pkl")
    record(environment, agent, f"{path}/{penalty}.mp4", max_steps=ENVIRONMENT_MAX_STEPS)
