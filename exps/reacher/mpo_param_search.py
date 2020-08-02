"""Run Reacher MB-MPO Hparam search."""

from lsf_runner import init_runner, make_commands

runner = init_runner(f"GPUCRL_Reacher_mbmpo", num_threads=1)

cmd_list = make_commands(
    "mbmpo.py",
    base_args={"exploration": "expected"},
    fixed_hyper_args={},
    common_hyper_args={
        "mpo-num-iter": [100, 200],
        "mpo-eta": [0.5, 1.0, 1.5],
        "mpo-eta-mean": [0.7, 1.0, 1.3],
        "mpo-eta-var": [0.1, 1.0, 5.0],
    },
    algorithm_hyper_args={},
)
runner.run(cmd_list)
