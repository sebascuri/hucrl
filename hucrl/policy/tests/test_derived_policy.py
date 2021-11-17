import pytest
import torch
from rllib.policy import DerivedPolicy, NNPolicy
from rllib.util.distributions import Delta
from rllib.util.neural_networks.utilities import random_tensor
from rllib.util.utilities import tensor_to_distribution
from torch.distributions import MultivariateNormal


@pytest.fixture(params=[True, False])
def discrete_state(request):
    return request.param


@pytest.fixture(params=[True, False])
def discrete_action(request):
    return request.param


@pytest.fixture(params=[1, 4])
def dim_state(request):
    return request.param


@pytest.fixture(params=[1, 4])
def dim_action(request):
    return request.param


@pytest.fixture(params=[None, 1, 4])
def batch_size(request):
    return request.param


@pytest.fixture(params=[True, False])
def deterministic(request):
    return request.param


class TestDerivedPolicy(object):
    def init(
        self,
        discrete_state,
        discrete_action,
        dim_state,
        dim_action,
        deterministic=False,
        goal=None,
    ):

        self.num_states, self.dim_state = (
            (dim_state, ()) if discrete_state else (-1, (dim_state,))
        )

        self.num_actions, self.dim_action = (
            (dim_action, ()) if discrete_action else (-1, (dim_action,))
        )

        if discrete_state:
            base_dim = 1
        else:
            base_dim = self.dim_state[0]

        if discrete_action:
            base_dim += 1
        else:
            base_dim += self.dim_action[0]

        base_policy = NNPolicy(
            dim_state=self.dim_state,
            dim_action=(base_dim,),
            num_states=self.num_states,
            num_actions=self.num_actions,
            layers=[32, 32],
            deterministic=deterministic,
            goal=goal,
        )

        self.policy = DerivedPolicy(base_policy, self.dim_action)

    def test_raise_error_at_initialization(
        self, discrete_state, discrete_action, dim_state, dim_action
    ):
        if discrete_state or discrete_action:
            with pytest.raises(NotImplementedError):
                self.init(discrete_state, discrete_action, dim_state, dim_action)

    def test_property_values(self, dim_state, dim_action):
        self.init(False, False, dim_state, dim_action)
        assert self.policy.dim_action == (dim_action,)
        assert self.policy.dim_state == (dim_state,)
        assert not self.policy.discrete_action
        assert not self.policy.discrete_state

    def test_random_action(self, dim_state, dim_action):
        self.init(False, False, dim_state, dim_action)

        distribution = tensor_to_distribution(self.policy.random())
        sample = distribution.sample()

        assert distribution.mean.shape == self.dim_action
        assert sample.shape == (dim_action,)

    def test_forward(self, dim_state, dim_action, batch_size, deterministic):
        self.init(False, False, dim_state, dim_action, deterministic)
        state = random_tensor(False, dim_state, batch_size)
        distribution = tensor_to_distribution(self.policy(state))
        sample = distribution.sample()

        if deterministic:
            assert isinstance(distribution, Delta)
        else:
            assert isinstance(distribution, MultivariateNormal)

        if batch_size:
            assert distribution.mean.shape == (batch_size,) + self.dim_action
            if not deterministic:
                assert distribution.covariance_matrix.shape == (
                    batch_size,
                    self.dim_action[0],
                    self.dim_action[0],
                )
            assert sample.shape == (batch_size, dim_action)
        else:
            assert distribution.mean.shape == self.dim_action
            if not deterministic:
                assert distribution.covariance_matrix.shape == (
                    self.dim_action[0],
                    self.dim_action[0],
                )
            assert sample.shape == torch.Size((dim_action,))

    def test_goal(self, batch_size):
        goal = random_tensor(False, 3, None)
        self.init(False, False, 4, 2, goal=goal)
        state = random_tensor(False, 4, batch_size)
        pi = tensor_to_distribution(self.policy(state))
        action = pi.sample()
        assert action.shape == torch.Size([batch_size, 2] if batch_size else [2])
        assert action.dtype is torch.get_default_dtype()

        other_goal = random_tensor(False, 3, None)
        self.policy.set_goal(other_goal)
        other_pi = tensor_to_distribution(self.policy(state))

        assert not torch.any(other_pi.mean == pi.mean)
