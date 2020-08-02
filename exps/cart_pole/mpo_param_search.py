"""Run Cart Pole MB-MPO Hparam search."""

from lsf_runner import init_runner, make_commands

runner = init_runner(f"GPUCRL_CartPole_mbmpo", num_threads=1)

cmd_list = make_commands(
    "mbmpo.py",
    base_args={"exploration": "expected"},
    fixed_hyper_args={},
    common_hyper_args={
        "plan-horizon": [0, 1, 4],
        "mpo-num-iter": [50, 100, 200],
        "mpo-eta": [0.5, 1.0, 1.5],
        "mpo-eta-mean": [0.7, 1.0, 1.3],
        "mpo-eta-var": [0.1, 1.0, 5.0],
        "sim-initial-states-num-trajectories": [8],
        "sim-initial-dist-num-trajectories": [16, 32],
    },
    algorithm_hyper_args={},
)
runner.run(cmd_list)
