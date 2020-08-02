"""Implementation of an Optimistic Model."""
import torch
from rllib.model.transformed_model import TransformedModel


class OptimisticModel(TransformedModel):
    """Optimistic Model returns a Delta at the optimistic next state."""

    def __init__(self, base_model, transformations, beta=1.0):
        super().__init__(base_model, transformations)
        self.dim_action = (self.dim_action[0] + self.dim_state[0],)
        self.beta = beta

    def forward(self, state, action):
        """Get Optimistic Next state."""
        control_action = action[..., : -self.dim_state[0]]
        optimism_vars = action[..., -self.dim_state[0] :]
        optimism_vars = torch.clamp(optimism_vars, -1.0, 1.0)

        mean, tril = self.next_state(state, control_action)
        return (
            mean + self.beta * (tril @ optimism_vars.unsqueeze(-1)).squeeze(-1),
            torch.zeros_like(tril),
        )  # , tril)

    def scale(self, state, action):
        """Get scale at current state-action pair."""
        control_action = action[..., : -self.dim_state[0]]
        scale = super().scale(state, control_action)

        return scale
