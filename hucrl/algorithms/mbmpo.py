"""Maximum a Posterior Policy Optimization algorithm."""

import torch
import torch.distributions
from rllib.algorithms.abstract_algorithm import AbstractAlgorithm
from rllib.algorithms.mpo import MPOWorker
from rllib.dataset.datatypes import Loss
from rllib.util.neural_networks import (
    deep_copy_module,
    freeze_parameters,
    update_parameters,
)
from rllib.util.utilities import RewardTransformer, separated_kl, tensor_to_distribution
from rllib.util.value_estimation import mb_return


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
        old_policy = deep_copy_module(policy)
        freeze_parameters(old_policy)

        super().__init__(policy=policy, critic=value_function, gamma=gamma)
        self.old_policy = old_policy
        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.policy = policy
        self.value_function = value_function
        self.value_target = deep_copy_module(value_function)

        self.mpo_loss = MPOWorker(epsilon, epsilon_mean, epsilon_var, regularization)
        self.value_loss = criterion(reduction="mean")

        self.num_action_samples = num_action_samples
        self.reward_transformer = reward_transformer
        self.termination_model = termination_model
        self.dist_params = {}

    def forward(self, observation):
        """Compute the losses for one step of MPO.

        Note to future self: MPO uses the reversed mode-seeking KL-divergence.
        Don't change the next direction of the KL divergence.

        TRPO/PPO use the forward mean-seeking KL-divergence.
        kl_mean, kl_var = separated_kl(p=pi_dist_old, q=pi_dist)

        Parameters
        ----------
        observation : Observation
            The states at which to compute the losses.

        Returns
        -------
        loss : torch.Tensor
            The combined loss
        value_loss : torch.Tensor
            The loss of the value function approximation.
        policy_loss : torch.Tensor
            The kl-regularized fitting loss for the policy.
        eta_loss : torch.Tensor
            The loss for the lagrange multipliers.
        kl_mean : torch.Tensor
            The average KL divergence of the mean.
        kl_var : torch.Tensor
            The average KL divergence of the variance.
        """
        states = observation.state
        value_prediction = self.value_function(states)

        pi_dist = tensor_to_distribution(self.policy(states))
        pi_dist_old = tensor_to_distribution(self.old_policy(states))
        kl_mean, kl_var = separated_kl(p=pi_dist_old, q=pi_dist)

        with torch.no_grad():
            value_estimate, trajectory = mb_return(
                state=states,
                dynamical_model=self.dynamical_model,
                policy=self.old_policy,
                reward_model=self.reward_model,
                num_steps=1,
                gamma=self.gamma,
                value_function=self.value_target,
                num_samples=self.num_action_samples,
                reward_transformer=self.reward_transformer,
                termination_model=self.termination_model,
                **self.dist_params,
            )
        q_values = value_estimate
        action_log_p = pi_dist.log_prob(trajectory[0].action / self.policy.action_scale)

        # Since actions come from policy, value is the expected q-value
        mpo_loss = self.mpo_loss(
            q_values=q_values, action_log_p=action_log_p, kl_mean=kl_mean, kl_var=kl_var
        )

        value_loss = self.value_loss(value_prediction, q_values.mean(dim=0))
        td_error = value_prediction - q_values.mean(dim=0)

        self._info = {
            "kl_div": kl_mean + kl_var,
            "kl_mean": kl_mean,
            "kl_var": kl_var,
            "eta": self.mpo_loss.eta(),
            "eta_mean": self.mpo_loss.eta_mean(),
            "eta_var": self.mpo_loss.eta_var(),
        }

        return mpo_loss + Loss(critic_loss=value_loss, td_error=td_error)

    def reset(self):
        """Reset the optimization (kl divergence) for the next epoch."""
        # Copy over old policy for KL divergence
        self.old_policy.load_state_dict(self.policy.state_dict())

    def update(self):
        """Update target value function."""
        update_parameters(
            self.value_target, self.value_function, tau=self.value_function.tau
        )
