"""Plotters for GP-UCRL experiments."""
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib import rcParams
from rllib.dataset.utilities import stack_list_of_tuples

LABELS = OrderedDict(
    optimistic=r"\textbf{H-UCRL}", expected="Greedy", thompson="Thompson"
)

COLORS = {"thompson": "C2", "expected": "C3", "optimistic": "C0"}


def set_figure_params(serif=False, fontsize=9):
    """Define default values for font, fontsize and use latex.

    Parameters
    ----------
    serif: bool, optional
        Whether to use a serif or sans-serif font.
    fontsize: int, optional
        Size to use for the font size

    """
    params = {
        "font.serif": [
            "Times",
            "Palatino",
            "New Century Schoolbook",
            "Bookman",
            "Computer Modern Roman",
        ]
        + rcParams["font.serif"],
        "font.sans-serif": [
            "Times",
            "Helvetica",
            "Avant Garde",
            "Computer Modern Sans serif",
        ]
        + rcParams["font.sans-serif"],
        "font.family": "sans-serif",
        "text.usetex": True,
        # Make sure mathcal doesn't use the Times style
        #  'text.latex.preamble':
        # r'\DeclareMathAlphabet{\mathcal}{OMS}{cmsy}{m}{n}',
        "axes.labelsize": fontsize,
        "axes.linewidth": 0.75,
        "font.size": fontsize,
        "legend.fontsize": fontsize * 0.7,
        "xtick.labelsize": fontsize * 8 / 9,
        "ytick.labelsize": fontsize * 8 / 9,
        "figure.dpi": 100,
        "savefig.dpi": 600,
        "legend.numpoints": 1,
    }

    if serif:
        params["font.family"] = "serif"

    rcParams.update(params)


def plot_last_trajectory(agent, environment, episode: int):
    """Plot agent last trajectory."""
    model = agent.dynamical_model.base_model
    real_trajectory = stack_list_of_tuples(agent.last_trajectory)

    for transformation in agent.dataset.transformations:
        real_trajectory = transformation(real_trajectory)

    fig, axes = plt.subplots(
        model.dim_state[0] + model.dim_action[0] + 1, 1, sharex="col"
    )

    for i in range(model.dim_state[0]):
        axes[i].plot(real_trajectory.state[:, i])
        axes[i].set_ylabel(f"State {i}")

    for i in range(model.dim_action[0]):
        axes[model.dim_state[0] + i].plot(real_trajectory.action[:, i])
        axes[model.dim_state[0] + i].set_ylabel(f"Action {i}")

    axes[-1].plot(real_trajectory.reward)
    axes[-1].set_ylabel(f"Reward")
    axes[-1].set_xlabel("Time")

    img_name = f"{agent.comment.title()}"
    plt.suptitle(f"{img_name} Episode {episode + 1}", y=1)

    if "DISPLAY" in os.environ:
        plt.draw()
    else:
        plt.savefig(f"{agent.logger.log_dir}/{episode + 1}.png")
    plt.close()


def plot_last_sim_and_real_trajectory(agent, environment, episode: int):
    """Plot agent last simulated and real trajectory."""
    model = agent.dynamical_model.base_model
    real_trajectory = stack_list_of_tuples(agent.last_trajectory)
    sim_trajectory = agent.sim_trajectory

    for transformation in agent.dataset.transformations:
        real_trajectory = transformation(real_trajectory)
        sim_trajectory = transformation(sim_trajectory)

    fig, axes = plt.subplots(
        model.dim_state[0] + model.dim_action[0] + 1, 2, sharex="col"
    )

    axes[0, 0].set_title("Real Trajectory")
    axes[0, 1].set_title("Sim Trajectory")

    for i in range(model.dim_state[0]):
        axes[i, 0].plot(real_trajectory.state[:, i])
        axes[i, 1].plot(sim_trajectory.state[:, 0, 0, i])

        axes[i, 0].set_ylabel(f"State {i}")

    for i in range(model.dim_action[0]):
        axes[model.dim_state[0] + i, 0].plot(real_trajectory.action[:, i])
        axes[model.dim_state[0] + i, 1].plot(sim_trajectory.action[:, 0, 0, i])
        axes[model.dim_state[0] + i, 0].set_ylabel(f"Action {i}")

    axes[-1, 0].plot(real_trajectory.reward)
    axes[-1, 1].plot(sim_trajectory.reward[:, 0, 0])

    axes[-1, 0].set_ylabel(f"Reward")
    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")

    img_name = f"{agent.comment.title()}"
    plt.suptitle(f"{img_name} Episode {episode + 1}", y=1)

    if "DISPLAY" in os.environ:
        plt.draw()
    else:
        plt.savefig(f"{agent.logger.log_dir}/{episode + 1}.png")

    plt.close()


def plot_last_rewards(agent, environment, episode: int):
    """Plot agent last rewards."""
    real_trajectory = stack_list_of_tuples(agent.last_trajectory)

    fig, axes = plt.subplots(1, 1, sharex="col")

    axes.plot(real_trajectory.reward)
    axes.set_ylabel("Reward")
    axes.set_xlabel("Time")

    img_name = f"{agent.comment.title()}"
    plt.suptitle(f"{img_name} Episode {episode + 1}", y=1)

    if "DISPLAY" in os.environ:
        plt.show()
    else:
        plt.savefig(f"{agent.logger.log_dir}/{episode + 1}.png")

    plt.close()
