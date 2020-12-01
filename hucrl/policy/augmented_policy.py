"""Python Script Template."""
import torch
from rllib.policy import AbstractPolicy, NNPolicy


class AugmentedPolicy(AbstractPolicy):
    """Augmented Policy class.

    An augmented policy maintains a true and a halluciantion policy and it stacks the
    predictions.
    """

    def __init__(self, true_policy, hallucination_policy, *args, **kwargs):
        dim_state = true_policy.dim_state
        dim_action = (true_policy.dim_action[0] + hallucination_policy.dim_action[0],)
        action_scale = torch.cat(
            (true_policy.action_scale, hallucination_policy.action_scale), dim=-1
        )
        super().__init__(
            dim_state=dim_state,
            dim_action=dim_action,
            action_scale=action_scale,
            goal=true_policy.goal,
            dist_params=true_policy.dist_params,
            *args,
            **kwargs,
        )
        self.true_policy = true_policy
        self.hallucination_policy = hallucination_policy

    def forward(self, state):
        """Forward compute the policy."""
        p_mean, p_scale_tril = self.true_policy(state)
        h_mean, h_scale_tril = self.hallucination_policy(state)

        p_std = p_scale_tril.diagonal(dim1=-1, dim2=-2)
        h_std = h_scale_tril.diagonal(dim1=-1, dim2=-2)
        mean = torch.cat((p_mean, h_mean), dim=-1)[..., : self.dim_action[0]]
        std = torch.cat((p_std, h_std), dim=-1)[..., : self.dim_action[0]]
        return mean, std.diag_embed()

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See AbstractPolicy.default()."""
        true_policy = NNPolicy.default(environment, *args, **kwargs)
        hallucination_policy = NNPolicy.default(
            environment, dim_action=environment.dim_state, *args, **kwargs
        )
        hallucination_policy.action_scale = torch.ones(environment.dim_state)
        return cls(
            true_policy=true_policy,
            hallucination_policy=hallucination_policy,
            *args,
            **kwargs,
        )

    @property
    def deterministic(self):
        """Return if policy is deterministic."""
        return self._deterministic

    @deterministic.setter
    def deterministic(self, value):
        """Set flag if the policy is deterministic or not."""
        self.true_policy.deterministic = value
        self._deterministic = value

    @torch.jit.export
    def reset(self):
        """Reset policy parameters (for example internal states)."""
        self.true_policy.reset()
        self.hallucination_policy.reset()

    @torch.jit.export
    def update(self):
        """Update policy parameters."""
        self.true_policy.update()
        self.hallucination_policy.update()

    @torch.jit.export
    def set_goal(self, goal=None):
        """Set policy goal."""
        self.true_policy.set_goal(goal)
        self.hallucination_policy.set_goal(goal)
