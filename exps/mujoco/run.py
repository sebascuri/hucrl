"""Script that demonstrates how to use BPTT using hallucination."""

import argparse
import importlib

from rllib.environment import GymEnvironment
from rllib.model import TransformedModel
from rllib.util import set_random_seed
from rllib.util.training.agent_training import evaluate_agent, train_agent

from exps.util import parse_config_file
from hucrl.environment.hallucination_wrapper import HallucinationWrapper
from hucrl.model.hallucinated_model import HallucinatedModel


def main(args):
    """Run experiment."""
    set_random_seed(args.seed)
    env_config = parse_config_file(args.env_config_file)

    environment = GymEnvironment(
        env_config["name"], ctrl_cost_weight=env_config["action_cost"], seed=args.seed
    )
    reward_model = environment.env.reward_model()
    if args.exploration == "optimistic":
        dynamical_model = HallucinatedModel.default(environment, beta=args.beta)
        environment.add_wrapper(HallucinationWrapper)
    else:
        dynamical_model = TransformedModel.default(environment)
    kwargs = parse_config_file(args.agent_config_file)

    agent = getattr(
        importlib.import_module("rllib.agent"), f"{args.agent}Agent"
    ).default(
        environment=environment,
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        thompson_sampling=args.exploration == "thompson",
        **kwargs,
    )
    train_agent(
        agent=agent,
        environment=environment,
        max_steps=env_config["max_steps"],
        num_episodes=args.train_episodes,
        render=args.render,
        print_frequency=1,
    )

    evaluate_agent(
        agent=agent,
        environment=environment,
        max_steps=env_config["max_steps"],
        num_episodes=args.test_episodes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for H-UCRL.")
    parser.add_argument(
        "--agent",
        type=str,
        default="BPTT",
        choices=["BPTT", "MVE", "DataAugmentation", "MPC", "MBMPO"],
    )
    parser.add_argument("--agent-config-file", type=str, default="config/bptt.yaml")
    parser.add_argument(
        "--env-config-file", type=str, default="config/envs/half-cheetah.yaml"
    )

    parser.add_argument(
        "--exploration",
        type=str,
        default="optimistic",
        choices=["optimistic", "expected", "thompson"],
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-episodes", type=int, default=250)
    parser.add_argument("--test-episodes", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=1)

    parser.add_argument("--render", action="store_true", default=False)

    parser.add_argument("--beta", type=float, default=1.0)
    main(parser.parse_args())
