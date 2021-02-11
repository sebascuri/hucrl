"""Maximum a Posterior Policy Optimization algorithm."""

import torch
import torch.distributions
from rllib.algorithms.abstract_algorithm import AbstractAlgorithm
from rllib.algorithms.mpo import MPOLoss
from rllib.dataset.datatypes import Loss
from rllib.util.utilities import RewardTransformer

from hucrl.utilities import mb_return


class MBMPO(AbstractAlgorithm):
    """Model-Based MPO Algorithm.

    This method uses the `MPOLoss`, but constructs the Q-function using the value
    function together with the model.

    Parameters
    ----------
    dynamical_model : AbstractModel
    reward_model : AbstractReward
    policy : AbstractPolicy
    value_function : AbstractValueFunction
    epsilon : float
        The average KL-divergence for the E-step, i.e., the KL divergence between
        the sample-based policy and the original policy
    epsilon_mean : float
        The KL-divergence for the mean component in the M-step (fitting the policy).
        See `mbrl.control.separated_kl`.
    epsilon_var : float
        The KL-divergence for the variance component in the M-step (fitting the policy).
        See `mbrl.control.separated_kl`.
    gamma : float
        The discount factor.
    """

    def __init__(
        self,
        dynamical_model,
        reward_model,
        policy,
        value_function,
        criterion,
        epsilon=0.1,
        epsilon_mean=0.0,
        epsilon_var=0.0001,
        regularization=False,
        gamma=0.99,
        num_action_samples=15,
        reward_transformer=RewardTransformer(),
        termination_model=None,
    ):
        super().__init__(
            policy=policy,
            criterion=criterion(reduction="mean"),
            critic=value_function,
            epsilon_mean=epsilon_mean,
            epsilon_var=epsilon_var,
            regularization=regularization,
            num_samples=num_action_samples,
            reward_transformer=reward_transformer,
            gamma=gamma,
        )
        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.termination_model = termination_model
        self.mpo_loss = MPOLoss(epsilon, regularization)

    def critic_loss(self, observation):
        """Return 0 loss. The actor loss returns both the critic and the actor."""
        return Loss()

    def actor_loss(self, observation):
        """Compute the losses for one step of MPO.

        Parameters
        ----------
        observation : Observation
            The states at which to compute the losses.
        """
        state = observation.state
        value_prediction = self.critic(state)

        with torch.no_grad():
            value_estimate, obs = mb_return(
                state=state,
                dynamical_model=self.dynamical_model,
                policy=self.old_policy,
                reward_model=self.reward_model,
                num_steps=1,
                gamma=self.gamma,
                value_function=self.critic_target,
                num_samples=self.num_samples,
                reward_transformer=self.reward_transformer,
                termination_model=self.termination_model,
                reduction="min",
            )
        q_values = value_estimate
        log_p, _ = self.get_log_p_and_ope_weight(obs.state, obs.action)

        # Since actions come from policy, value is the expected q-value
        mpo_loss = self.mpo_loss(q_values=q_values, action_log_p=log_p.squeeze(-1))
        value_loss = self.criterion(value_prediction, q_values.mean(dim=0))
        td_error = value_prediction - q_values.mean(dim=0)

        critic_loss = Loss(critic_loss=value_loss, td_error=td_error)
        self._info.update(eta=self.mpo_loss.eta)
        return mpo_loss.reduce(self.criterion.reduction) + critic_loss
