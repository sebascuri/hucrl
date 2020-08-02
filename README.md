# rllib

[![CircleCI](https://img.shields.io/circleci/build/github/sebascuri/rllib/master?label=master%20build%20and%20test&token=25c056fd6b7e322c55dd48fd0c6052b1f8800919)](https://app.circleci.com/pipelines/github/sebascuri/rllib)
[![CircleCI](https://img.shields.io/circleci/build/github/sebascuri/rllib/dev?label=dev%20build%20and%20test&token=25c056fd6b7e322c55dd48fd0c6052b1f8800919)](https://app.circleci.com/pipelines/github/sebascuri/rllib)
[![CircleCI](https://circleci.com/gh/sebascuri/rllib/tree/master.svg?style=shield&circle-token=25c056fd6b7e322c55dd48fd0c6052b1f8800919)](https://circleci.com/gh/circleci/circleci-docs/tree/teesloane-patch-5)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://pypi.org/project/hug/)


To install create a conda environment:
```bash
$ conda create -n rllib python=3.7
$ conda activate rllib
```

```bash
$ pip install -e .[test,logging,experiments]
```

For Mujoco (license required) Run:
```bash
$ pip install -e .[mujoco]
```

On clusters run:
```bash
$ sudo apt-get install -y --no-install-recommends --quiet build-essential libopenblas-dev python-opengl xvfb xauth
```


## Running an experiment.
```bash
$ python exps/run $ENVIRONMENT $AGENT
```

For help, see
```bash
$ python exps/run.py --help
```

## Pre Commit
install pre-commit with
```bash
$ pip install pre-commit
$ pre-commit install
```

Run pre-commit with
```bash
$ pre-commit run --all-files
```


## CIRCLE-CI

To run locally circleci run:
```bash
$ circleci config process .circleci/config.yml > process.yml
$ circleci local execute -c process.yml --job test
```

## Goals
Environment goals are passed to the agent through agent.set_goal(goal).
If a goal moves during an episode, then include it in the observation space of the environment.
If a goal is to follow a trajectory, it might be a good idea to encode it in the reward model.

## Policies
Continuous Policies are "bounded" between [-1, 1] via a tanh transform unless otherwise defined.
For environments with action spaces with different bounds, up(down)-scale the action after sampling it.
