"""Run MBMPO on Pusher."""
from dotmap import DotMap

from exps.mb_mpo_arguments import parser
from exps.plotters import plot_last_rewards
from exps.pusher import (
    ACTION_COST,
    ENVIRONMENT_MAX_STEPS,
    TRAIN_EPISODES,
    get_agent_and_environment,
)
from exps.util import train_and_evaluate

PLAN_HORIZON = 4
PLAN_SAMPLES = 500
MPO_ETA, MPO_ETA_MEAN, MPO_ETA_VAR = 0.5, 0.1, 0.5
MPO_NUM_ITER = 50
SIM_TRAJECTORIES = 64
SIM_EXP_TRAJECTORIES = 0  # 32
SIM_MEMORY_TRAJECTORIES = 0  # 8
SIM_NUM_STEPS = ENVIRONMENT_MAX_STEPS

parser.description = "Run Pusher using Model-Based MPO."
parser.set_defaults(
    # exploration='thompson',
    action_cost=ACTION_COST,
    train_episodes=TRAIN_EPISODES,
    environment_max_steps=ENVIRONMENT_MAX_STEPS,
    plan_horizon=PLAN_HORIZON,
    plan_samples=PLAN_SAMPLES,
    mpo_num_iter=MPO_NUM_ITER,
    # mpo_eta=.5,
    # mpo_eta_mean=1.,
    # mpo_eta_var=5.,
    mpo_eta=None,
    mpo_eta_mean=None,
    mpo_eta_var=None,
    mpo_epsilon=0.1,
    mpo_epsilon_mean=0.1,
    mpo_epsilon_var=1e-4,
    sim_num_steps=SIM_NUM_STEPS,
    sim_initial_states_num_trajectories=SIM_TRAJECTORIES,
    sim_initial_dist_num_trajectories=SIM_EXP_TRAJECTORIES,
    sim_memory_num_trajectories=SIM_MEMORY_TRAJECTORIES,
    model_kind="DeterministicEnsemble",
    model_learn_num_iter=5,
    max_memory=ENVIRONMENT_MAX_STEPS,
    model_layers=[200, 200, 200],
    model_non_linearity="swish",
    model_opt_lr=1e-4,
    model_opt_weight_decay=0.0005,
    mpo_opt_lr=5e-4,
    mpo_gradient_steps=50,
    policy_layers=[100, 100],
    value_function_layers=[200, 200],
)

args = parser.parse_args()
params = DotMap(vars(args))

environment, agent = get_agent_and_environment(params, "mbmpo")
agent.mpo_target_update_frequency = 20
agent.dynamical_model.beta = 5.0
# agent.exploration_episodes = 10
train_and_evaluate(
    agent, environment, params=params, plot_callbacks=[plot_last_rewards]
)
