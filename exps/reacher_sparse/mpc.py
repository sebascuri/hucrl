"""Run Reacher-Sparse with MPC agent."""

from dotmap import DotMap

from exps.mpc_arguments import parser
from exps.reacher_sparse import (
    ACTION_COST,
    ENVIRONMENT_MAX_STEPS,
    TRAIN_EPISODES,
    get_agent_and_environment,
)
from exps.util import train_and_evaluate

MPC_HORIZON, MPC_NUM_SAMPLES = 20, 400

parser.description = "Run Sparse Reacher using Model-Based MPC."
parser.set_defaults(
    exploration="expected",
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
)

args = parser.parse_args()
params = DotMap(vars(args))

# shallow has action, length_scale = (1., .4)
# lazy has action, length_scale = (1., .3)

environment, agent = get_agent_and_environment(params, "mpc")

environment.env.action_scale = 1.0
agent.reward_model.action_scale = 1.0
environment.env.length_scale = 0.3
agent.reward_model.length_scale = 0.3
# print(environment.env.action_scale)
# print(environment.env.length_scale)
# print(agent.reward_model.action_scale)
# print(agent.reward_model.length_scale)

train_and_evaluate(agent, environment, params=params, plot_callbacks=[])
