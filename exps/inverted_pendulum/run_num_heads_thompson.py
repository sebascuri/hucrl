"""Run optimistic exploration experiments."""
from typing import Dict, List

from lsf_runner import init_runner, make_commands

from exps.inverted_pendulum import ACTION_COST

#  multimodularity.

nn_ensemble_hyper_params = {
    "model-kind": ["ProbabilisticEnsemble"],
    "model-learn-num-iter": [50],
    "model-num-heads": [50, 100],
}  # type: Dict[str, List]

for algorithm_hyper_args in [nn_ensemble_hyper_params]:
    runner = init_runner(
        f"GPUCRL_Inverted_Pendulum_{algorithm_hyper_args['model-kind'][0]}",
        num_threads=1,
        num_workers=27,
    )

    cmd_list = make_commands(
        "mbmpo.py",
        base_args={"num-threads": 1},
        fixed_hyper_args={},
        common_hyper_args={
            "seed": [0, 1, 2, 3, 4],
            "exploration": ["thompson"],
            "action-cost": [0, ACTION_COST, 2 * ACTION_COST],
        },
        algorithm_hyper_args=algorithm_hyper_args,
    )

    runner.run_batch(cmd_list)
