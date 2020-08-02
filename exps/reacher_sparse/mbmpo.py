"""Run reacher sparse with MBMPO."""
from dotmap import DotMap
from rllib.util.utilities import RewardTransformer

from exps.mb_mpo_arguments import parser
from exps.plotters import plot_last_rewards
from exps.reacher_sparse import (
    ACTION_COST,
    ENVIRONMENT_MAX_STEPS,
    TRAIN_EPISODES,
    get_agent_and_environment,
)
from exps.util import train_and_evaluate

PLAN_HORIZON = 30
PLAN_SAMPLES = 500
PLAN_ELITES = 20
ALGORITHM_NUM_ITER = 10
SIM_TRAJECTORIES = 50
SIM_EXP_TRAJECTORIES = 150
SIM_MEMORY_TRAJECTORIES = 80
SIM_NUM_STEPS = 5
SIM_SUBSAMPLE = 1

parser.description = "Run Reacher using Model-Based MPO."
parser.set_defaults(
    # exploration='expected',
    action_cost=1 * ACTION_COST,
    train_episodes=TRAIN_EPISODES,
    environment_max_steps=ENVIRONMENT_MAX_STEPS,
    plan_horizon=PLAN_HORIZON,
    plan_samples=PLAN_SAMPLES,
    plan_elites=PLAN_ELITES,
    mpo_num_iter=ALGORITHM_NUM_ITER,
    # mpo_eta=1.,
    # mpo_eta_mean=1.,
    # mpo_eta_var=5.,
    mpo_eta=None,
    mpo_eta_mean=None,
    mpo_eta_var=None,
    mpo_epsilon=0.1,
    mpo_epsilon_mean=0.01,
    mpo_epsilon_var=0.0001,
    mpo_opt_lr=1e-4,
    mpo_batch_size=100,
    mpo_gradient_steps=30,
    mpo_target_frequency_update=1,
    sim_num_steps=SIM_NUM_STEPS,
    sim_initial_states_num_trajectories=SIM_TRAJECTORIES,
    sim_initial_dist_num_trajectories=SIM_EXP_TRAJECTORIES,
    sim_memory_num_trajectories=SIM_MEMORY_TRAJECTORIES,
    model_kind="ProbabilisticEnsemble",
    model_learn_num_iter=50,
    max_memory=10 * ENVIRONMENT_MAX_STEPS,
    model_layers=[64, 64],
    model_non_linearity="swish",
    model_opt_lr=1e-4,
    model_opt_weight_decay=0.0005,
    policy_layers=[100, 100],
    policy_tau=0.005,
    policy_non_linearity="ReLU",
    value_function_layers=[200, 200],
    value_function_tau=0.005,
    value_function_non_linearity="ReLU",
)

args = parser.parse_args()
params = DotMap(vars(args))

environment, agent = get_agent_and_environment(params, "mbmpo")

# All tasks have rewards that are scaled to be between 0 and 1000.
agent.algorithm.reward_transformer = RewardTransformer(offset=0, scale=500)
# agent.exploration_episodes = 3
train_and_evaluate(
    agent, environment, params=params, plot_callbacks=[plot_last_rewards]
)
