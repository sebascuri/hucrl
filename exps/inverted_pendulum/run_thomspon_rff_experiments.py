"""Run optimistic exploration experiments."""

from lsf_runner import init_runner, make_commands

from exps.inverted_pendulum import ACTION_COST

runner = init_runner(f"GPUCRL_Inverted_Pendulum_rff", num_threads=1, num_workers=21)

cmd_list = make_commands(
    "mbmpo.py",
    base_args={"num-threads": 1, "model-kind": "FeatureGP", "model-learn-num-iter": 0},
    fixed_hyper_args={},
    common_hyper_args={
        "seed": [0, 1, 2, 3, 4],
        "exploration": ["thompson"],
        "action-cost": [0, ACTION_COST, 2 * ACTION_COST],
        "model-num-features": [256, 625, 1296],
        "model-feature-approximation": ["QFF", "RFF"],
    },
    algorithm_hyper_args={},
)

runner.run_batch(cmd_list)
