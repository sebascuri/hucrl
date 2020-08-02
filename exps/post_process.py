"""Post-Processing utilities."""
import json
import os

import pandas as pd


def parse_results(base_dir, agent):
    """Parse all results from base directory.

    Parameters
    ----------
    base_dir: str.
        Relative path to base directory.
    agent: str.
        Name of agent.

    Examples
    --------
    parse_results('runs/Cartpoleenv', 'MBMPO'))

    """
    log_dirs = os.listdir(f"{base_dir}/{agent}Agent/")

    df = pd.DataFrame()
    for i, log_dir in enumerate(log_dirs):
        try:
            with open(f"{base_dir}/{agent}Agent/{log_dir}/hparams.json", "r") as f:
                params = json.load(f)
        except (
            json.JSONDecodeError,
            FileNotFoundError,
        ):  # If experiment did not finish, just continue.
            continue

        for key, value in params.items():
            if isinstance(value, list):
                params[key] = ",".join(str(s) for s in value)

        if "exploration" not in params:
            params["exploration"] = "optimistic" if params["optimistic"] else "expected"

        params["agent"] = agent
        params = pd.DataFrame(params, index=(0,))
        params["id"] = i

        try:
            with open(f"{base_dir}/{agent}Agent/{log_dir}/statistics.json", "r") as f:
                statistics = pd.read_json(f)
        except (
            json.JSONDecodeError,
            FileNotFoundError,
        ):  # If experiment did not finish, just continue.
            continue

        statistics["best_return"] = statistics["environment_return"].cummax()
        try:
            statistics["best_model_return"] = statistics["model_return"].cummax()
        except KeyError:
            pass
        statistics["id"] = i
        statistics["episode"] = statistics.index

        exp = pd.merge(statistics, params, on="id")

        df = pd.concat((df, exp), sort=False)
    return df


def print_df(
    df,
    idx=None,
    sort_key="best_return",
    keep="first",
    group_keys=("action_cost", "exploration", "model_kind"),
    print_keys=("best_return", "environment_return", "id"),
):
    """Print data frame by grouping and sorting data.

    It will group the data frame by group_keys in order and then print, per group,
    the maximum of the best_key.

    Parameters
    ----------
    df: pd.DataFrame.
        Data frame to sort and print.
    idx: int, optional.
        Time index in which to filter results.
    sort_key: str, optional.
        Key to sort by.
    keep: str, optional.
        Keep order in sorting. By default first.
    group_keys: Iter[str]
        Tuple of strings to group.
    print_keys: str:
        Tuple of strings to print.

    """
    max_idx = df.index.unique().max()
    idx = idx if idx is not None else max_idx
    idx = min(idx, max_idx)
    df = df[df.index == idx]
    df = df.sort_values(sort_key).drop_duplicates(list(group_keys), keep=keep)
    df = df.groupby(list(group_keys)).max()
    print(df[list(print_keys)])


def parse_statistics(base_dir, agent):
    """Parse statistics from base directory.

    Parameters
    ----------
    base_dir: str.
        Relative path to base directory.
    agent: str.
        Name of agent.

    Examples
    --------
    parse_results('runs/Cartpoleenv', 'MBMPO'))

    """
    log_dirs = os.listdir(f"{base_dir}/{agent}Agent/")

    df = pd.DataFrame()
    for i, log_dir in enumerate(log_dirs):
        with open(f"{base_dir}/{agent}Agent/{log_dir}/statistics.json", "r") as f:
            statistics = pd.read_json(f)
        statistics["best_return"] = statistics.loc[:, "environment_return"].cummax()
        statistics["id"] = i

        df = pd.concat((df, statistics))
    return df
