"""Run optimistic exploration experiments with NN ensembles."""
from typing import Dict, List

from lsf_runner import init_runner, make_commands

nn_ensemble_hyper_params = {
    "model-kind": ["ProbabilisticEnsemble", "DeterministicEnsemble"],
    "model-learn-num-iter": [50],
}  # type: Dict[str, List]

for algorithm_hyper_args in [nn_ensemble_hyper_params]:
    runner = init_runner(
        f"GPUCRL_Inverted_Pendulum_{algorithm_hyper_args['model-kind'][0]}",
        num_threads=1,
        num_workers=45,
    )

    cmd_list = make_commands(
        "mbmpo.py",
        base_args={"num-threads": 1},
        fixed_hyper_args={},
        common_hyper_args={
            "seed": [0, 1, 2, 3, 4],
            "exploration": ["expected", "optimistic", "thompson"],
            "action-cost": [0, 0.1, 0.2],
        },
        algorithm_hyper_args=algorithm_hyper_args,
    )

    runner.run_batch(cmd_list)
