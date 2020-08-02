"""Run Half-Cheetah with MBMPO agent."""

from dotmap import DotMap

from exps.half_cheetah import (
    ACTION_COST,
    ENVIRONMENT_MAX_STEPS,
    TRAIN_EPISODES,
    get_agent_and_environment,
)
from exps.mb_mpo_arguments import parser
from exps.plotters import plot_last_rewards
from exps.util import train_and_evaluate

PLAN_HORIZON = 4
MPO_ETA, MPO_ETA_MEAN, MPO_ETA_VAR = 1.0, 1.0, 5.0
MPO_NUM_ITER = 100
SIM_TRAJECTORIES = 4
SIM_EXP_TRAJECTORIES = 32
SIM_NUM_STEPS = 200

parser.description = "Run Half-Cheetah using Model-Based MPO."
parser.set_defaults(
    action_cost=ACTION_COST,
    exploration="expected",
    train_episodes=TRAIN_EPISODES,
    environment_max_steps=ENVIRONMENT_MAX_STEPS,
    plan_horizon=PLAN_HORIZON,
    mpo_num_iter=MPO_NUM_ITER,
    mpo_eta=MPO_ETA,
    mpo_eta_mean=MPO_ETA_MEAN,
    mpo_eta_var=MPO_ETA_VAR,
    sim_num_steps=SIM_NUM_STEPS,
    sim_initial_states_num_trajectories=SIM_TRAJECTORIES,
    sim_initial_dist_num_trajectories=SIM_EXP_TRAJECTORIES,
    model_kind="ProbabilisticEnsemble",
    model_learn_num_iter=25,
    max_memory=ENVIRONMENT_MAX_STEPS,
    model_layers=[100, 100, 100, 100],
    model_non_linearity="swish",
    model_opt_lr=1e-4,
    mpo_opt_lr=1e-5,
)


args = parser.parse_args()
params = DotMap(vars(args))

environment, agent = get_agent_and_environment(params, "mbmpo")
train_and_evaluate(agent, environment, params, plot_callbacks=[plot_last_rewards])
