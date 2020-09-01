import copy
import os

import pytest
from rllib.agent import MPCAgent
from rllib.algorithms.mpc import CEMShooting, MPPIShooting, RandomShooting
from rllib.dataset.experience_replay import ExperienceReplay
from rllib.environment import GymEnvironment
from rllib.model.environment_model import EnvironmentModel
from rllib.util.training import evaluate_agent

SEED = 0
MAX_ITER = 5
ENVIRONMENT = "VContinuous-CartPole-v0"

env = GymEnvironment(ENVIRONMENT, SEED)
env_model = copy.deepcopy(env)
env_model.reset()
dynamical_model = EnvironmentModel(env_model)
reward_model = EnvironmentModel(env_model, model_kind="rewards")
termination = EnvironmentModel(env_model, model_kind="termination")
GAMMA = 0.99
HORIZON = 5
NUM_ITER = 5
NUM_SAMPLES = 50
NUM_ELITES = 5
KAPPA = 1.0
BETAS = [0.2, 0.8, 0]

memory = ExperienceReplay(max_len=2000, num_steps=1)

value_function = None


@pytest.fixture(params=["random_shooting", "cem_shooting", "mppi_shooting"])
def solver(request):
    return request.param


@pytest.fixture(params=[True, False])
def warm_start(request):
    return request.param


@pytest.fixture(params=["mean", "zero", "constant"])
def default_action(request):
    return request.param


@pytest.fixture(params=[1])
def num_cpu(request):
    return request.param


def get_solver(solver_, warm_start_, num_cpu_, default_action_):
    if solver_ == "random_shooting":
        mpc_solver = RandomShooting(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            horizon=HORIZON,
            gamma=1.0,
            num_samples=NUM_SAMPLES,
            num_elites=NUM_ELITES,
            termination=termination,
            terminal_reward=value_function,
            warm_start=warm_start_,
            default_action=default_action_,
            num_cpu=num_cpu_,
        )
    elif solver_ == "cem_shooting":
        mpc_solver = CEMShooting(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            horizon=HORIZON,
            gamma=1.0,
            num_iter=NUM_ITER,
            num_samples=NUM_SAMPLES,
            num_elites=NUM_ELITES,
            termination=termination,
            terminal_reward=value_function,
            warm_start=warm_start_,
            default_action=default_action_,
            num_cpu=num_cpu_,
        )
    elif solver_ == "mppi_shooting":
        mpc_solver = MPPIShooting(
            dynamical_model=dynamical_model,
            reward_model=reward_model,
            horizon=HORIZON,
            gamma=1.0,
            num_iter=NUM_ITER,
            kappa=KAPPA,
            filter_coefficients=BETAS,
            num_samples=NUM_SAMPLES,
            termination=termination,
            terminal_reward=value_function,
            warm_start=warm_start_,
            default_action=default_action_,
            num_cpu=num_cpu_,
        )
    else:
        raise NotImplementedError
    return mpc_solver


def test_mpc_solvers(solver, num_cpu):
    if num_cpu > 1 and "CI" in os.environ:
        return
    mpc_solver = get_solver(solver, True, num_cpu, "mean")

    agent = MPCAgent(mpc_solver=mpc_solver)
    evaluate_agent(
        agent, environment=env, num_episodes=1, max_steps=MAX_ITER, render=False
    )


def test_mpc_warm_start(solver, warm_start):
    mpc_solver = get_solver(solver, warm_start, 1, "mean")

    agent = MPCAgent(mpc_solver=mpc_solver)
    evaluate_agent(
        agent, environment=env, num_episodes=1, max_steps=MAX_ITER, render=False
    )


def test_mpc_default_action(solver, default_action):
    mpc_solver = get_solver(solver, True, 1, default_action)

    agent = MPCAgent(mpc_solver=mpc_solver)
    evaluate_agent(
        agent, environment=env, num_episodes=1, max_steps=MAX_ITER, render=False
    )
