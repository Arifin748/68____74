# Copyright 2024 OmniSafe Team. All Rights Reserved.
#
# Adam-RCPO implementation
# Modified from RCPO to update Lagrange multiplier with Adam optimizer

from __future__ import annotations

import torch

from omnisafe.algorithms.on_policy.naive_lagrange.rcpo import RCPO
from omnisafe.common.adam_lagrange import AdamLagrange


class AdamRCPO(RCPO):
    """Adam-RCPO Algorithm.

    This algorithm is a variant of RCPO where the Lagrange multiplier
    is updated using Adam optimizer instead of standard gradient ascent.
    """

    def _init(self) -> None:
        """Initialize Adam Lagrange multiplier."""
        super()._init()

        self._lagrange = AdamLagrange(
            cost_limit=self._cfgs.algo_cfgs.cost_limit,
            lr=self._cfgs.algo_cfgs.lagrange_lr,
        )

    def _update(self) -> None:
        """Update actor, critic, and Adam Lagrange multiplier."""
        super()._update()

        # update lagrange multiplier using episode cost
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        self._lagrange.update_lagrange_multiplier(ep_cost)

        # log lambda
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrange_multiplier.item(),
            },
        )