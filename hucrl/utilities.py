"""Utilities to estimate value functions."""
from collections import namedtuple

from rllib.dataset.utilities import stack_list_of_tuples
from rllib.util.neural_networks.utilities import repeat_along_dimension
from rllib.util.rollout import rollout_model
from rllib.util.utilities import RewardTransformer
from rllib.util.value_estimation import n_step_return

MBValueReturn = namedtuple("MBValueReturn", ["value_estimate", "trajectory"])


def mc_return(
    observation,
    gamma=1.0,
    lambda_=1.0,
    reward_transformer=RewardTransformer(),
    value_function=None,
    reduction="none",
    entropy_regularization=0.0,
):
    r"""Calculate n-step MC return from the trajectory.

    The N-step return of a trajectory is calculated as:
    .. math:: V(s) = \sum_{t=0}^T \gamma^t (r + \lambda H) + \gamma^{T+1} V(s_{T+1}).

    Parameters
    ----------
    observation: Observation
        List of observations to compute the n-step return.
    gamma: float, optional.
        Discount factor.
    lambda_: float, optional.
        Lambda return.
    value_function: AbstractValueFunction, optional.
        Value function to bootstrap the value of the final state.
    entropy_regularization: float, optional.
        Entropy regularization coefficient.
    reward_transformer: RewardTransformer
    reduction: str.
        How to reduce ensemble value functions.

    """
    if observation.reward.ndim == 0 or len(observation.reward) == 0:
        return 0.0
    returns = n_step_return(
        observation,
        gamma=gamma,
        reward_transformer=reward_transformer,
        entropy_regularization=entropy_regularization,
        value_function=value_function,
        reduction=reduction,
    )
    return returns[..., -1]


def mb_return(
    state,
    dynamical_model,
    reward_model,
    policy,
    num_steps=1,
    gamma=1.0,
    value_function=None,
    num_samples=1,
    entropy_reg=0.0,
    reward_transformer=RewardTransformer(),
    termination_model=None,
    reduction="none",
):
    r"""Estimate the value of a state by propagating the state with a model for N-steps.

    Rolls out the model for a number of `steps` and sums up the rewards. After this,
    it bootstraps using the value function. With T = steps:

    .. math:: V(s) = \sum_{t=0}^T \gamma^t r(s, \pi(s)) + \gamma^{T+1} V(s_{T+1})

    Note that `steps=0` means that the model is still used to predict the next state.

    Parameters
    ----------
    state: torch.Tensor
        Initial state from which planning starts. It accepts a batch of initial states.
    dynamical_model: AbstractModel
        The model predicts a distribution over next states given states and actions.
    reward_model: AbstractReward
        The reward predicts a distribution over floats or ints given states and actions.
    policy: AbstractPolicy
        The policy predicts a distribution over actions given the state.
    num_steps: int, optional. (default=1).
        Number of steps predicted with the model before (optionally) bootstrapping.
    gamma: float, optional. (default=1.).
        Discount factor.
    value_function: AbstractValueFunction, optional. (default=None).
        The value function used for bootstrapping, takes states as input.
    num_samples: int, optional. (default=0).
        The states are repeated `num_repeats` times in order to estimate the expected
        value by MC sampling of the policy, rewards and dynamics (jointly).
    entropy_reg: float, optional. (default=0).
        Entropy regularization parameter.
    termination_model: AbstractModel, optional. (default=None).
        Callable that returns True if the transition yields a terminal state.
    reward_transformer: RewardTransformer.

    Returns
    -------
    return: DynaReturn
        q_target:
            Num_samples of MC estimation of q-function target.
        trajectory:
            Sample trajectory that MC estimation produces.

    References
    ----------
    Lowrey, K., Rajeswaran, A., Kakade, S., Todorov, E., & Mordatch, I. (2018).
    Plan online, learn offline: Efficient learning and exploration via model-based
    control. ICLR.

    Sutton, R. S. (1991).
    Dyna, an integrated architecture for learning, planning, and reacting. ACM.

    Silver, D., Sutton, R. S., & Müller, M. (2008).
    Sample-based learning and search with permanent and transient memories. ICML.
    """
    # Repeat states to get a better estimate of the expected value
    state = repeat_along_dimension(state, number=num_samples, dim=0)
    trajectory = rollout_model(
        dynamical_model=dynamical_model,
        reward_model=reward_model,
        policy=policy,
        initial_state=state,
        max_steps=num_steps,
        termination_model=termination_model,
    )
    observation = stack_list_of_tuples(trajectory, dim=state.ndim - 1)
    value = mc_return(
        observation=observation,
        gamma=gamma,
        value_function=value_function,
        entropy_regularization=entropy_reg,
        reward_transformer=reward_transformer,
        reduction=reduction,
    )

    return MBValueReturn(value, observation)
