"""Plotters for gp_ucrl pendulum experiments."""
import itertools
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams
from rllib.dataset.utilities import stack_list_of_tuples
from rllib.model.gp_model import ExactGPModel
from rllib.util.utilities import moving_average_filter

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


def combinations(arrays):
    """Return a single array with combinations of parameters.

    Parameters
    ----------
    arrays : list of np.array

    Returns
    -------
    array : np.array
        An array that contains all combinations of the input arrays
    """
    return np.array(np.meshgrid(*arrays)).T.reshape(-1, len(arrays))


def linearly_spaced_combinations(bounds, num_entries):
    """
    Return 2-D array with all linearly spaced combinations with the bounds.

    Parameters
    ----------
    bounds : sequence of tuples
        The bounds for the variables, [(x1_min, x1_max), (x2_min, x2_max), ...]
    num_entries : integer or array_like
        Number of samples to use for every dimension. Can be a constant if
        the same number should be used for all, or an array to fine-tune
        precision. Total number of data points is num_samples ** len(bounds).

    Returns
    -------
    combinations : 2-d array
        A 2-d arrray. If d = len(bounds) and l = prod(num_samples) then it
        is of size l x d, that is, every row contains one combination of
        inputs.
    """
    bounds = np.atleast_2d(bounds)
    num_vars = len(bounds)
    num_entries = np.broadcast_to(num_entries, num_vars)

    # Create linearly spaced test inputs
    inputs = [np.linspace(b[0], b[1], n) for b, n in zip(bounds, num_entries)]

    # Convert to 2-D array
    return combinations(inputs)


def plot_combinations_as_grid(axis, values, num_entries, bounds=None, **kwargs):
    """Take values from a grid and plot them as an image.

    Takes values generated from `linearly_spaced_combinations`.

    Parameters
    ----------
    axis: matplotlib.AxesSubplot.
    values: ndarray.
    num_entries: array_like.
        Number of samples to use for every dimension.
        Used for reshaping
    bounds: sequence of tuples.
        The bounds for the variables, [(x1_min, x1_max), (x2_min, x2_max), ...]
    kwargs: dict.
        Passed to axis.imshow
    """
    kwargs["origin"] = "lower"
    if bounds is not None:
        kwargs["extent"] = list(itertools.chain(*bounds))
    return axis.imshow(values.reshape(*num_entries).T, **kwargs)


def plot_on_grid(function, bounds, num_entries, axis):
    """Plot function values on a grid.

    Parameters
    ----------
    function: callable.
    bounds: list.
    num_entries: list.
    axis: matplotlib.AxesSubplot.


    Returns
    -------
    axis
    """
    states = linearly_spaced_combinations(bounds, num_entries)
    values = function(torch.tensor(states, dtype=torch.get_default_dtype()))
    values = values.detach().numpy()

    img = plot_combinations_as_grid(axis, values, num_entries, bounds)
    plt.colorbar(img, ax=axis)
    axis.set_xlim(bounds[0])
    axis.set_ylim(bounds[1])
    return axis


def plot_state_trajectory(state, axis):
    """Plot state trajectory."""
    if state.dim() == 4:
        axis.plot(state[:, 0, 0, 0], state[:, 0, 0, 1], color="C1")
        axis.plot(state[-1, 0, 0, 0], state[-1, 0, 0, 1], "x", color="C1")
    else:
        axis.plot(state[:, 0], state[:, 1], color="C1")
        axis.plot(state[-1, 0], state[-1, 1], "x", color="C1")


def plot_learning_losses(policy_losses, value_losses, horizon):
    """Plot the losses encountnered during learning.

    Parameters
    ----------
    policy_losses : list or ndarray
    value_losses : list or ndarray
    horizon : int
        Horizon used for smoothing
    """
    t = np.arange(len(policy_losses))

    fig, axes = plt.subplots(ncols=2, figsize=(15, 5))

    axes[0].plot(t, policy_losses)
    axes[0].plot(*moving_average_filter(t, policy_losses, horizon), label="smoothed")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Policy loss")
    axes[0].legend()

    axes[1].plot(t, value_losses)
    axes[1].plot(*moving_average_filter(t, value_losses, horizon), label="smoothed")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Value loss")
    axes[1].legend()

    if "DISPLAY" in os.environ:
        plt.show()


def plot_trajectory_states_and_rewards(states, rewards):
    """Plot the states and rewards from a trajectory."""
    fig, axes = plt.subplots(ncols=2, figsize=(15, 5))

    axes[0].plot(states[:, 0, 0], states[:, 0, 1], "x")
    axes[0].plot(states[-1, 0, 0], states[-1, 0, 1], "x")
    axes[0].set_xlabel("Angle [rad]")
    axes[0].set_ylabel("Angular velocity [rad/s]")

    axes[1].plot(rewards[:, 0, 0])
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Instantaneous reward")

    if "DISPLAY" in os.environ:
        plt.show()


def plot_values_and_policy(
    value_function, policy, bounds, num_entries, trajectory=None, suptitle=None
):
    """Plot the value and policy function over a grid.

    Parameters
    ----------
    value_function : torch.nn.Module
    policy : torch.nn.Module
    bounds : list
    num_entries : list
    trajectory : Observation
    suptitle : str, optional.

    """
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey="col", figsize=(20, 8))

    plot_on_grid(value_function, bounds=bounds, num_entries=num_entries, axis=axes[0])
    if trajectory is not None:
        plot_state_trajectory(trajectory.state, axes[0])
    plot_on_grid(
        lambda x: policy(x)[0], bounds=bounds, num_entries=num_entries, axis=axes[1]
    )
    if trajectory is not None:
        plot_state_trajectory(trajectory.state, axes[1])
    axes[0].set_title("Value function")
    axes[1].set_title("Policy")

    for ax in axes:
        ax.set_xlabel("Angle[rad]")
        ax.set_ylabel("Angular velocity [rad/s]")
        ax.axis("tight")
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])

    if suptitle:
        plt.suptitle(suptitle, y=1)

    if "DISPLAY" in os.environ:
        plt.show()


def plot_returns_entropy_kl(returns, entropy, kl_div):
    """Plot returns, entropy and KL Divergence."""
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(np.arange(len(returns)), returns)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Returns")

    axes[1].plot(np.arange(len(entropy)), entropy)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Entropy")

    axes[2].plot(np.arange(len(kl_div)), kl_div)
    axes[2].set_xlabel("Iteration")

    if "DISPLAY" in os.environ:
        plt.show()


def plot_pendulum_trajectories(agent, environment, episode: int):
    """Plot GP inputs and trajectory in a Pendulum environment."""
    model = agent.dynamical_model.base_model
    trajectory = stack_list_of_tuples(agent.last_trajectory)
    sim_obs = agent.sim_trajectory

    for transformation in agent.dataset.transformations:
        trajectory = transformation(trajectory)
        sim_obs = transformation(sim_obs)
    if isinstance(model, ExactGPModel):
        fig, axes = plt.subplots(
            1 + model.dim_state[0] // 2, 2, sharex="col", sharey="row"
        )
    else:
        fig, axes = plt.subplots(1, 2, sharex="col", sharey="row")
        axes = axes[np.newaxis]
    fig.set_size_inches(5.5, 2.0)
    # Plot real trajectory
    sin, cos = torch.sin(trajectory.state[:, 0]), torch.cos(trajectory.state[:, 0])
    axes[0, 0].scatter(
        torch.atan2(sin, cos) * 180 / np.pi,
        trajectory.state[:, 1],
        c=trajectory.action[:, 0],
        cmap="jet",
        vmin=-1,
        vmax=1,
    )
    axes[0, 0].set_title("Real Trajectory")

    # Plot sim trajectory
    sin = torch.sin(sim_obs.state[:, 0, 0, 0])
    cos = torch.cos(sim_obs.state[:, 0, 0, 0])
    axes[0, 1].scatter(
        torch.atan2(sin, cos) * 180 / np.pi,
        sim_obs.state[:, 0, 0, 1],
        c=sim_obs.action[:, 0, 0, 0],
        cmap="jet",
        vmin=-1,
        vmax=1,
    )
    axes[0, 1].set_title("Optimistic Trajectory")

    if isinstance(model, ExactGPModel):
        for i in range(model.dim_state[0]):
            inputs = model.gp[i].train_inputs[0]
            sin, cos = inputs[:, 1], inputs[:, 0]
            axes[1 + i // 2, i % 2].scatter(
                torch.atan2(sin, cos) * 180 / np.pi,
                inputs[:, 2],
                c=inputs[:, 3],
                cmap="jet",
                vmin=-1,
                vmax=1,
            )
            axes[1 + i // 2, i % 2].set_title(f"GP {i} data.")

            if hasattr(model.gp[i], "xu"):
                inducing_points = model.gp[i].xu
                sin, cos = inducing_points[:, 1], inducing_points[:, 0]
                axes[1 + i // 2, i % 2].scatter(
                    torch.atan2(sin, cos) * 180 / np.pi,
                    inducing_points[:, 2],
                    c=inducing_points[:, 3],
                    cmap="jet",
                    marker="*",
                    vmin=-1,
                    vmax=1,
                )

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlim([-180, 180])
            ax.set_ylim([-15, 15])

    for i in range(axes.shape[0]):
        axes[i, 0].set_ylabel("Angular Velocity [rad/s]")

    for j in range(axes.shape[1]):
        axes[-1, j].set_xlabel("Angle [degree]")

    # img_name = f"{agent.comment.title()}"
    if "optimistic" in agent.comment.lower():
        name = "H-UCRL"
    elif "expected" in agent.comment.lower():
        name = "Greedy"
    elif "thompson" in agent.comment.lower():
        name = "Thompson"
    else:
        raise NotImplementedError
    plt.suptitle(f"{name} Episode {episode + 1}", x=0.53, y=0.96)

    plt.tight_layout()
    plt.savefig(f"{agent.logger.log_dir}/{episode + 1}.pdf")

    if "DISPLAY" in os.environ:
        plt.show()
    plt.close(fig)
