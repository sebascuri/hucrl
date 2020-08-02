"""Run optimistic exploration experiments with GP models."""
from typing import Dict, List

from lsf_runner import init_runner, make_commands

gp_hyper_params = {
    "model-kind": ["ExactGP"],
    "model-max-num-points": [int(1e10), 200],
}  # type: Dict[str, List]

# sparse_gp_hyper_params = {
#     'model-kind': ['SparseGP'],
#     'model-max-num-points': [int(1e10), 200],
#     'model-sparse-approximation': ['DTC', 'FITC'],
# }
#
# features_gp_hyper_params = {
#     'model-kind': ['FeatureGP'],
#     'model-num-features': [256, 625, 1296],
#     'model-feature-approximation': ['QFF', 'RFF'],
# }

for algorithm_hyper_args in [gp_hyper_params]:
    # sparse_gp_hyper_params,
    # features_gp_hyper_params]:
    runner = init_runner(
        f"GPUCRL_Inverted_Pendulum_{algorithm_hyper_args['model-kind'][0]}",
        num_threads=1,
        num_workers=12,
    )

    cmd_list = make_commands(
        "mbmpo.py",
        base_args={"num-threads": 1},
        fixed_hyper_args={},
        common_hyper_args={
            "seed": [0, 1, 2, 3, 4],
            "exploration": ["expected", "optimistic"],
            "action-cost": [0, 0.1, 0.2, 0.5],
        },
        algorithm_hyper_args=algorithm_hyper_args,
    )

    runner.run_batch(cmd_list)
