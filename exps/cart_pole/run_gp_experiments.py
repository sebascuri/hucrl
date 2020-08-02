"""Run optimistic exploration experiments."""

from lsf_runner import init_runner, make_commands

from exps.cart_pole import ACTION_COST

gp_hyper_params = {
    "model-kind": ["ExactGP"],
    "model-max-num-points": [int(1e10), 200],
    "model_learn_num_iter": [0],
}

for algorithm_hyper_args in [gp_hyper_params]:
    for agent in ["mpc", "mbmpo"]:
        runner = init_runner(
            f"GPUCRL_CartPole_{algorithm_hyper_args['model-kind']}_{agent}",
            num_threads=1,
            num_workers=8,
        )

        cmd_list = make_commands(
            f"{agent}.py",
            base_args={"num-threads": 1},
            fixed_hyper_args={},
            common_hyper_args={
                "seed": [0, 1, 2, 3, 4],
                "exploration": ["expected", "optimistic"],
                "action-cost": [0, ACTION_COST, 2 * ACTION_COST, 5 * ACTION_COST],
            },
            algorithm_hyper_args=algorithm_hyper_args,
        )

        runner.run_batch(cmd_list)
