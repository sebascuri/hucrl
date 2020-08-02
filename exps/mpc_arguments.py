"""Command line arguments for MPC Agents."""
import argparse

parser = argparse.ArgumentParser(description="Parameters for MPC.")

parser.add_argument(
    "--exploration",
    type=str,
    default="optimistic",
    choices=["optimistic", "expected", "thompson"],
)
parser.add_argument(
    "--seed", type=int, default=0, help="initial random seed (default: 0)."
)
parser.add_argument("--train-episodes", type=int, default=10)
parser.add_argument("--test-episodes", type=int, default=1)
parser.add_argument("--num-threads", type=int, default=1)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--max-memory", type=int, default=10000)

environment_parser = parser.add_argument_group("environment")
environment_parser.add_argument("--action-cost", type=float, default=0.2)
environment_parser.add_argument("--gamma", type=float, default=0.99)
environment_parser.add_argument("--environment-max-steps", type=int, default=400)

model_parser = parser.add_argument_group("model")
model_parser.add_argument(
    "--model-kind",
    type=str,
    default="ExactGP",
    choices=[
        "ExactGP",
        "SparseGP",
        "FeatureGP",
        "ProbabilisticNN",
        "DeterministicNN",
        "ProbabilisticEnsemble",
        "DeterministicEnsemble",
    ],
)
model_parser.add_argument("--model-learn-num-iter", type=int, default=0)
model_parser.add_argument("--model-learn-batch-size", type=int, default=32)
model_parser.add_argument("--not-bootstrap", action="store_true")

model_parser.add_argument(
    "--model-sparse-approximation",
    type=str,
    default="DTC",
    choices=["DTC", "SOR", "FITC"],
)
model_parser.add_argument(
    "--model-feature-approximation",
    type=str,
    default="QFF",
    choices=["QFF", "RFF", "OFF"],
)
model_parser.add_argument("--model-opt-lr", type=float, default=5e-4)
model_parser.add_argument("--model-opt-weight-decay", type=float, default=0)
model_parser.add_argument("--model-max-num-points", type=int, default=int(1e10))
model_parser.add_argument("--model-sparse-q-bar", type=int, default=2)
model_parser.add_argument("--model-num-features", type=int, default=625)
model_parser.add_argument("--model-layers", type=int, nargs="*", default=[64, 64])
model_parser.add_argument("--model-non-linearity", type=str, default="ReLU")
model_parser.add_argument("--model-unbiased-head", action="store_false")
model_parser.add_argument("--model-num-heads", type=int, default=5)


value_function_parser = parser.add_argument_group("value function")
value_function_parser.add_argument(
    "--value-function-layers", type=int, nargs="*", default=[64, 64]
)
value_function_parser.add_argument(
    "--value-function-non-linearity", type=str, default="ReLU"
)
value_function_parser.add_argument(
    "--value-function-unbiased-head", action="store_false"
)

value_opt_parser = parser.add_argument_group("policy_evaluation")
value_opt_parser.add_argument("--value-opt-lr", type=float, default=5e-4)
value_opt_parser.add_argument("--value-opt-weight-decay", type=float, default=0)
value_opt_parser.add_argument("--value-opt-num-iter", type=int, default=0)
value_opt_parser.add_argument("--value-opt-batch-size", type=int, default=32)
value_opt_parser.add_argument("--value-gradient-steps", type=int, default=50)
value_opt_parser.add_argument("--value-num-steps-returns", type=int, default=1)


mpc_parser = parser.add_argument_group("mpc")
mpc_parser.add_argument(
    "--mpc-solver", type=str, default="cem", choices=["cem", "random", "mppi"]
)
mpc_parser.add_argument("--mpc-horizon", type=int, default=20)
mpc_parser.add_argument("--mpc-num-iter", type=int, default=5)
mpc_parser.add_argument("--mpc-num-samples", type=int, default=400)
mpc_parser.add_argument("--mpc-num-elites", type=int, default=40)
mpc_parser.add_argument("--mpc-alpha", type=float, default=0.1)

mpc_parser.add_argument("--mpc-terminal-reward", action="store_true")
mpc_parser.add_argument("--mpc-not-warm-start", action="store_true")
mpc_parser.add_argument(
    "--mpc-default-action",
    type=str,
    default="zero",
    choices=["constant", "mean", "zero"],
)
mpc_parser.add_argument("--mpc-kappa", type=float, default=1.0)
mpc_parser.add_argument(
    "--mpc-filter-coefficients", type=float, nargs="+", default=[0.2, 0.8, 0]
)


sim_parser = parser.add_argument_group("simulation")
sim_parser.add_argument("--sim-num-steps", type=int, default=0)
sim_parser.add_argument("--sim-initial-states-num-trajectories", type=int, default=0)
sim_parser.add_argument("--sim-initial-dist-num-trajectories", type=int, default=0)
sim_parser.add_argument("--sim_memory-num-trajectories", type=int, default=0)
sim_parser.add_argument("--sim-num-subsample", type=int, default=1)

viz_parser = parser.add_argument_group("visualization")
viz_parser.add_argument("--plot-train-results", action="store_true")
viz_parser.add_argument("--render-train", action="store_true")
viz_parser.add_argument("--render-test", action="store_true")
viz_parser.add_argument("--print-frequency", type=int, default=1)
