"""Template for a Model Based Agent.

A model based agent has three behaviors:
- It learns models from data collected from the environment.
- It optimizes policies with simulated data from the models.
- It plans with the model and policies (as guiding sampler).
"""
import contextlib
from dataclasses import asdict

import gpytorch
import torch
from gym.utils import colorize
from rllib.agent.abstract_agent import AbstractAgent
from rllib.dataset.datatypes import Observation
from rllib.dataset.experience_replay import (
    BootstrapExperienceReplay,
    StateExperienceReplay,
)
from rllib.dataset.utilities import average_dataclass, stack_list_of_tuples
from rllib.model import ExactGPModel, TransformedModel
from rllib.util.gaussian_processes import SparseGP
from rllib.util.neural_networks.utilities import DisableGradient
from rllib.util.rollout import rollout_model
from rllib.util.training.model_learning import train_model
from rllib.util.utilities import tensor_to_distribution
from rllib.util.value_estimation import mb_return
from tqdm import tqdm


class ModelBasedAgent(AbstractAgent):
    """Implementation of a Model Based RL Agent.

    Parameters
    ----------
    dynamical_model: AbstractModel.
        Fixed or learnable dynamical model.
    reward_model: AbstractReward.
        Fixed or learnable reward model.
    model_optimizer: Optim
        Optimizer for dynamical_model and reward_model.
    policy: AbstractPolicy.
        Fixed or learnable policy.
    value_function: AbstractValueFunction, optional. (default: None).
        Fixed or learnable value function used for planning.
    termination_model: Callable, optional. (default: None).
        Fixed or learnable termination_model condition.

    plan_horizon: int, optional. (default: 0).
        If plan_horizon = 0: the agent returns a sample from the current policy when
        'agent.act(state)' is called.
        If plan_horizon > 0: the agent uses the model to plan for plan_horizon steps and
        returns the action that optimizes the plan.
    plan_samples: int, optional. (default: 1).
        Number of samples used to solve the planning problem.
    plan_elites: int, optional. (default: 1).
        Number of elite samples used to return the best action.

    model_learn_num_iter: int, optional. (default: 0).
        Number of iteration for model learning.
    model_learn_batch_size: int, optional. (default: 64).
        Batch size of model learning algorithm.
    max_memory: int, optional. (default: 10000).
        Maximum size of data set.

    policy_opt_num_iter: int, optional. (default: 0).
        Number of iterations for policy optimization.
    policy_opt_batch_size: int, optional. (default: model_learn_batch_size).
        Batch size of policy optimization algorithm.

    sim_num_steps: int, optional. (default: 20).
        Number of simulation steps.
    sim_initial_states_num_trajectories: int, optional. (default: 8).
        Number of simulation trajectories that start from a sample of the empirical
        distribution.
    sim_initial_dist_num_trajectories: int, optional. (default: 0).
        Number of simulation trajectories that start from a sample of a selected initial
        distribution.
    sim_memory_num_trajectories: int, optional. (default: 0).
        Number of simulation trajectories that start from a sample of the dataset.
    sim_refresh_interval: int, optional.
        Number of policy optimization steps.
    sim_num_subsample: int, optional. (default: 1).
        Add one out of `sim_num_subsample' samples to the data set.
    initial_distribution: Distribution, optional. (default: None).
        Initial state distribution.
    thompson_sampling: bool, optional. (default: False).
        Bool that indicates to use thompson's sampling.
    gamma: float, optional. (default: 0.99).
    exploration_steps: int, optional. (default: 0).
    exploration_episodes: int, optional. (default: 0).
    comment: str, optional. (default: '').
    """

    def __init__(
        self,
        dynamical_model,
        reward_model,
        model_optimizer,
        policy,
        value_function=None,
        termination_model=None,
        plan_horizon=0,
        plan_samples=1,
        plan_elites=1,
        model_learn_num_iter=0,
        model_learn_batch_size=64,
        bootstrap=True,
        max_memory=10000,
        policy_opt_num_iter=0,
        policy_opt_batch_size=None,
        policy_opt_gradient_steps=0,
        policy_opt_target_update_frequency=1,
        policy_update_frequency=1,
        optimizer=None,
        sim_num_steps=20,
        sim_initial_states_num_trajectories=8,
        sim_initial_dist_num_trajectories=0,
        sim_memory_num_trajectories=0,
        sim_max_memory=10000,
        sim_refresh_interval=1,
        sim_num_subsample=1,
        initial_distribution=None,
        thompson_sampling=False,
        gamma=1.0,
        exploration_steps=0,
        exploration_episodes=0,
        tensorboard=False,
        comment="",
    ):
        super().__init__(
            train_frequency=0,
            num_rollouts=0,
            policy_update_frequency=policy_update_frequency,
            gamma=gamma,
            exploration_steps=exploration_steps,
            exploration_episodes=exploration_episodes,
            tensorboard=tensorboard,
            comment=comment,
        )
        if not isinstance(dynamical_model, TransformedModel):
            dynamical_model = TransformedModel(dynamical_model, [])
        self.dynamical_model = dynamical_model
        self.reward_model = reward_model
        self.termination_model = termination_model
        self.model_optimizer = model_optimizer
        self.value_function = value_function

        self.model_learn_num_iter = model_learn_num_iter
        self.model_learn_batch_size = model_learn_batch_size

        self.policy = policy
        # self.policy = DerivedPolicy(policy,self.dynamical_model.base_model.dim_action)

        self.plan_horizon = plan_horizon
        self.plan_samples = plan_samples
        self.plan_elites = plan_elites

        if hasattr(dynamical_model.base_model, "num_heads"):
            num_heads = dynamical_model.base_model.num_heads
        else:
            num_heads = 1

        self.dataset = BootstrapExperienceReplay(
            max_len=max_memory,
            transformations=dynamical_model.forward_transformations,
            num_bootstraps=num_heads,
            bootstrap=bootstrap,
        )
        self.sim_dataset = StateExperienceReplay(
            max_len=sim_max_memory, dim_state=self.dynamical_model.dim_state
        )
        self.initial_states = StateExperienceReplay(
            max_len=sim_max_memory, dim_state=self.dynamical_model.dim_state
        )

        self.policy_opt_num_iter = policy_opt_num_iter
        if policy_opt_batch_size is None:  # set the same batch size as in model learn.
            policy_opt_batch_size = self.model_learn_batch_size
        self.policy_opt_batch_size = policy_opt_batch_size
        self.policy_opt_gradient_steps = policy_opt_gradient_steps
        self.policy_opt_target_update_frequency = policy_opt_target_update_frequency
        self.optimizer = optimizer

        self.sim_trajectory = None

        self.sim_num_steps = sim_num_steps
        self.sim_initial_states_num_trajectories = sim_initial_states_num_trajectories
        self.sim_initial_dist_num_trajectories = sim_initial_dist_num_trajectories
        self.sim_memory_num_trajectories = sim_memory_num_trajectories
        self.sim_refresh_interval = sim_refresh_interval
        self.sim_num_subsample = sim_num_subsample
        self.initial_distribution = initial_distribution
        self.new_episode = True
        self.thompson_sampling = thompson_sampling

        if self.thompson_sampling:
            self.dynamical_model.set_prediction_strategy("posterior")

        if hasattr(self.dynamical_model.base_model, "num_heads"):
            num_heads = self.dynamical_model.base_model.num_heads
        else:
            num_heads = 1

        layout = {
            "Model Training": {
                "average": [
                    "Multiline",
                    [f"average/model-{i}" for i in range(num_heads)]
                    + ["average/model_loss"],
                ]
            },
            "Policy Training": {
                "average": [
                    "Multiline",
                    ["average/value_loss", "average/policy_loss", "average/eta_loss"],
                ]
            },
            "Returns": {
                "average": [
                    "Multiline",
                    ["average/environment_return", "average/model_return"],
                ]
            },
        }
        if self.logger.writer is not None:
            self.logger.writer.add_custom_scalars(layout)

    def act(self, state):
        """Ask the agent for an action to interact with the environment.

        If the plan horizon is zero, then it just samples an action from the policy.
        If the plan horizon > 0, then is plans with the current model.
        """
        if self.plan_horizon == 0:
            action = super().act(state)
        else:
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.get_default_dtype())
            policy = tensor_to_distribution(
                self.policy(state), **self.policy.dist_params
            )
            self.pi = policy
            action = self.plan(state).detach().numpy()

        # action = action[..., : self.dynamical_model.base_model.dim_action[0]]
        return action.clip(
            -self.policy.action_scale.numpy(), self.policy.action_scale.numpy()
        )

    def observe(self, observation):
        """Observe a new transition.

        If the episode is new, add the initial state to the state transitions.
        Add the transition to the data set.
        """
        super().observe(observation)
        self.dataset.append(observation)
        if self.new_episode:
            self.initial_states.append(observation.state.unsqueeze(0))
            self.new_episode = False

    def start_episode(self):
        """See `AbstractAgent.start_episode'."""
        super().start_episode()
        self.new_episode = True

        if self.thompson_sampling:
            self.dynamical_model.sample_posterior()

    def end_episode(self):
        """See `AbstractAgent.end_episode'.

        If the agent is training, and the base model is a GP Model, then add the
        transitions to the GP, and summarize and sparsify the GP Model.

        Then train the agent.
        """
        if self.training:
            if isinstance(self.dynamical_model.base_model, ExactGPModel):
                observation = stack_list_of_tuples(self.last_trajectory)
                for transform in self.dataset.transformations:
                    observation = transform(observation)
                print(colorize("Add data to GP Model", "yellow"))
                self.dynamical_model.base_model.add_data(
                    observation.state, observation.action, observation.next_state
                )

                print(colorize("Summarize GP Model", "yellow"))
                self.dynamical_model.base_model.summarize_gp()

                for i, gp in enumerate(self.dynamical_model.base_model.gp):
                    self.logger.update(**{f"gp{i} num inputs": len(gp.train_targets)})

                    if isinstance(gp, SparseGP):
                        self.logger.update(
                            **{f"gp{i} num inducing inputs": gp.xu.shape[0]}
                        )

            self.learn()
        super().end_episode()

    def plan(self, state):
        """Plan with current model and policy by (approximately) solving MPC.

        To solve MPC, the policy is sampled to guide random shooting.
        The average of the top `self.plan_elite' samples is returned.
        """
        self.dynamical_model.eval()
        value, trajectory = mb_return(
            state,
            dynamical_model=self.dynamical_model,
            reward_model=self.reward_model,
            policy=self.policy,
            num_steps=self.plan_horizon,
            gamma=self.gamma,
            num_samples=self.plan_samples,
            value_function=self.value_function,
            reward_transformer=self.algorithm.reward_transformer,
            termination_model=self.termination_model,
        )
        actions = stack_list_of_tuples(trajectory).action
        idx = torch.topk(value, k=self.plan_elites, largest=True)[1]
        # Return first action and the mean over the elite samples.
        return actions[0, idx].mean(0)

    def learn(self):
        """Train the agent.

        This consists of two steps:
            Step 1: Train Model with new data.
                Calls self.learn_model().
            Step 2: Optimize policy with simulated data.
                Calls self.simulate_and_learn_policy().
        """
        # Step 1: Train Model with new data.
        self.learn_model()
        if self.total_steps < self.exploration_steps or (
            self.total_episodes < self.exploration_episodes
        ):
            return

        # Step 2: Optimize policy with simulated data.
        self.simulate_and_learn_policy()

    def learn_model(self):
        """Train the models.

        This consists of different steps:
            Step 1: Train dynamical model.
            Step 2: TODO Train the reward model.
            Step 3: TODO Train the initial distribution model.
        """
        if self.model_learn_num_iter > 0:
            print(colorize("Training Model", "yellow"))

            train_model(
                self.dynamical_model.base_model,
                train_set=self.dataset,
                max_iter=self.model_learn_num_iter,
                optimizer=self.model_optimizer,
                logger=self.logger,
                batch_size=self.model_learn_batch_size,
                epsilon=-1.0,
            )

    def _log_simulated_trajectory(self):
        """Log simulated trajectory."""
        average_return = self.sim_trajectory.reward.sum(0).mean().item()
        average_scale = (
            torch.diagonal(self.sim_trajectory.next_state_scale_tril, dim1=-1, dim2=-2)
            .square()
            .sum(-1)
            .sum(0)
            .mean()
            .sqrt()
            .item()
        )
        self.logger.update(sim_entropy=self.sim_trajectory.entropy.mean().item())
        self.logger.update(sim_return=average_return)
        self.logger.update(sim_scale=average_scale)
        self.logger.update(sim_max_state=self.sim_trajectory.state.abs().max().item())
        self.logger.update(sim_max_action=self.sim_trajectory.action.abs().max().item())
        try:
            r_ctrl = self.reward_model.reward_ctrl.mean().detach().item()
            r_state = self.reward_model.reward_state.mean().detach().item()
            self.logger.update(sim_reward_ctrl=r_ctrl)
            self.logger.update(sim_reward_state=r_state)
        except AttributeError:
            pass
        try:
            r_o = self.reward_model.reward_dist_to_obj
            r_g = self.reward_model.reward_dist_to_goal
            self.logger.update(sim_reward_dist_to_obj=r_o.mean().detach().item())
            self.logger.update(sim_reward_dist_to_goal=r_g.mean().detach().item())
        except AttributeError:
            pass

    def simulate_and_learn_policy(self):
        """Simulate the model and optimize the policy with the learned data.

        This consists of two steps:
            Step 1: Simulate trajectories with the model.
                Calls self.simulate_model().
            Step 2: Implement a model free RL method that optimizes the policy.
                Calls self.learn_policy(). To be implemented by a Base Class.
        """
        print(colorize("Optimizing Policy with Model Data", "yellow"))
        self.dynamical_model.eval()
        self.sim_dataset.reset()  # Erase simulation data set before starting.
        with DisableGradient(self.dynamical_model), gpytorch.settings.fast_pred_var():
            for i in tqdm(range(self.policy_opt_num_iter)):
                # Step 1: Compute the state distribution
                with torch.no_grad():
                    self.simulate_model()

                # Log last simulations.
                self._log_simulated_trajectory()

                # Step 2: Optimize policy
                self.learn_policy()

                if (
                    self.sim_refresh_interval > 0
                    and (i + 1) % self.sim_refresh_interval == 0
                ):
                    self.sim_dataset.reset()

    def simulate_model(self):
        """Simulate the model.

        The simulation is initialized by concatenating samples from:
            - The empirical initial state distribution.
            - A learned or fixed initial state distribution.
            - The empirical state distribution.
        """
        # Samples from empirical initial state distribution.
        initial_states = self.initial_states.sample_batch(
            self.sim_initial_states_num_trajectories
        )

        # Samples from initial distribution.
        if self.sim_initial_dist_num_trajectories > 0:
            initial_states_ = self.initial_distribution.sample(
                (self.sim_initial_dist_num_trajectories,)
            )
            initial_states = torch.cat((initial_states, initial_states_), dim=0)

        # Samples from experience replay empirical distribution.
        if self.sim_memory_num_trajectories > 0:
            obs, *_ = self.dataset.sample_batch(self.sim_memory_num_trajectories)
            for transform in self.dataset.transformations:
                obs = transform.inverse(obs)
            initial_states_ = obs.state[:, 0, :]  # obs is an n-step return.
            initial_states = torch.cat((initial_states, initial_states_), dim=0)

        initial_states = initial_states.unsqueeze(0)
        self.policy.reset()
        trajectory = rollout_model(
            dynamical_model=self.dynamical_model,
            reward_model=self.reward_model,
            policy=self.policy,
            initial_state=initial_states,
            max_steps=self.sim_num_steps,
            termination_model=self.termination_model,
        )

        self.sim_trajectory = stack_list_of_tuples(trajectory)
        states = self.sim_trajectory.state.reshape(-1, *self.dynamical_model.dim_state)
        self.sim_dataset.append(states[:: self.sim_num_subsample])

    def learn_policy(self):
        """Optimize the policy."""
        # Iterate over state batches in the state distribution
        self.algorithm.reset()
        for _ in range(self.policy_opt_gradient_steps):

            def closure():
                """Gradient calculation."""
                states = Observation(
                    state=self.sim_dataset.sample_batch(self.policy_opt_batch_size)
                )
                self.optimizer.zero_grad()
                losses = self.algorithm(states)
                losses.combined_loss.backward()
                return losses

            if self.train_steps % self.policy_update_frequency == 0:
                cm = contextlib.nullcontext()
            else:
                cm = DisableGradient(self.policy)

            with cm:
                losses = self.optimizer.step(closure=closure)

            self.logger.update(**asdict(average_dataclass(losses)))
            self.logger.update(**self.algorithm.info())

            self.counters["train_steps"] += 1
            if self.train_steps % self.policy_opt_target_update_frequency == 0:
                self.algorithm.update()
                for param in self.params.values():
                    param.update()

            if self.early_stop(losses, **self.algorithm.info()):
                break

        self.algorithm.reset()
