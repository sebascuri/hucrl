"""Run Half-Cheetah experiments."""

from lsf_runner import init_runner, make_commands

from exps.half_cheetah import ACTION_COST

for agent in ["mpc", "mbmpo"]:
    runner = init_runner(
        f"GPUCRL_HalfCheetah_{agent}", num_threads=2, wall_time=1439, memory=2048
    )

    cmd_list = make_commands(
        f"{agent}.py",
        base_args={},
        fixed_hyper_args={},
        common_hyper_args={
            "exploration": ["thompson", "optimistic", "expected"],
            "model-kind": ["ProbabilisticEnsemble", "DeterministicEnsemble"],
            "action-cost": [0, ACTION_COST, 5 * ACTION_COST],
        },
        algorithm_hyper_args={},
    )
    runner.run(cmd_list)
