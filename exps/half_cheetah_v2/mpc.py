"""Run Half-Cheetah with MPC agent."""

from dotmap import DotMap

from exps.half_cheetah import (
    ACTION_COST,
    ENVIRONMENT_MAX_STEPS,
    TRAIN_EPISODES,
    get_agent_and_environment,
)
from exps.mpc_arguments import parser
from exps.plotters import plot_last_rewards
from exps.util import train_and_evaluate

MPC_HORIZON, MPC_NUM_SAMPLES = 30, 500

parser.description = "Run Half Cheetah using Model-Based MPC."
parser.set_defaults(
    action_cost=ACTION_COST,
    train_episodes=TRAIN_EPISODES,
    environment_max_steps=ENVIRONMENT_MAX_STEPS,
    mpc_horizon=MPC_HORIZON,
    mpc_num_samples=MPC_NUM_SAMPLES,
    mpc_num_elites=MPC_NUM_SAMPLES // 10,
    model_kind="ProbabilisticEnsemble",
    model_learn_num_iter=5,
    max_memory=10 * ENVIRONMENT_MAX_STEPS,
    model_layers=[256, 256, 256],
    model_non_linearity="swish",
    model_opt_lr=1e-4,
    model_opt_weight_decay=0.0005,
    num_threads=2,
)

args = parser.parse_args()
params = DotMap(vars(args))

environment, agent = get_agent_and_environment(params, "mpc")
train_and_evaluate(
    agent, environment, params=params, plot_callbacks=[plot_last_rewards]
)
