import torch


class AdamLagrange:
    """Lagrange multiplier optimized by Adam."""

    def __init__(self, cost_limit: float, lr: float = 0.01) -> None:
        self.cost_limit = cost_limit

        self.lagrange_multiplier = torch.nn.Parameter(
            torch.tensor(0.0),
        )

        self.optimizer = torch.optim.Adam(
            [self.lagrange_multiplier],
            lr=lr,
        )

    def update_lagrange_multiplier(self, ep_cost: float) -> None:
        """Update lambda using Adam."""
        self.optimizer.zero_grad()

        loss = -self.lagrange_multiplier * (ep_cost - self.cost_limit)

        loss.backward()

        self.optimizer.step()

        with torch.no_grad():
            self.lagrange_multiplier.clamp_(min=0.0)