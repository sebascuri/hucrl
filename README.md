# Implementation of H-UCRL Algorithm

[![CircleCI](https://img.shields.io/circleci/build/github/sebascuri/hucrl/master?label=master%20build%20and%20test&token=7f83bf4eea065c55015a2685c2b0ffbf996e3a2a)](https://app.circleci.com/pipelines/github/sebascuri/hucrl)
[![CircleCI](https://img.shields.io/circleci/build/github/sebascuri/hucrl/dev?label=dev%20build%20and%20test&token=7f83bf4eea065c55015a2685c2b0ffbf996e3a2a)](https://app.circleci.com/pipelines/github/sebascuri/hucrl)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://pypi.org/project/hug/)

This repository is an implementation of the H-UCRL algorithm introduced in
Curi, S., Berkenkamp, F., & Krause, A. (2020). Efficient Model-Based Reinforcement Learning through Optimistic Policy Search and Planning.



To install create a conda environment:
```bash
$ conda create -n hucrl python=3.7
$ conda activate hucrl
```

```bash
$ pip install -e .[test,logging,experiments]
```

For Mujoco (license required) Run:
```bash
$ pip install -e .[mujoco]
```


## Running an experiment.
For the inverted pendulum experiment run
```bash
$ python exps/inverted_pendulum/run.py
```

For the mujoco (license required) experiment run
```bash
$ python exps/mujoco/run.py --environment ENV_NAME --agent AGENT_NAME --action
```

We support MBHalfCheetah-v0, MBPusher-v0, MBReacher-v0, MBAnt-v0, MBCartPole-v0, MBHopper-v0,
MBInvertedDoublePendulum-v0, MBInvertedPendulum-v0, MBReacher-v0, MBReacher3D-v0, MBSwimmer-v0, MBWalker2d-v0

## Citing H-UCRL
If you this repo for your research please use the following BibTeX entry:
```text
@article{curi2020efficient,
  title={Efficient model-based reinforcement learning through optimistic policy search and planning},
  author={Curi, Sebastian and Berkenkamp, Felix and Krause, Andreas},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
