from __future__ import annotations

from omnisafe.algorithms.on_policy.naive_lagrange.rcpo import RCPO
from omnisafe.common.pid_lagrange import PIDLagrangian


class PIDRCPO(RCPO):
    """PID-RCPO Algorithm."""

    def _init(self) -> None:
        super()._init()

        self._lagrange = PIDLagrangian(
            pid_kp=self._cfgs.algo_cfgs.pid_kp,
            pid_ki=self._cfgs.algo_cfgs.pid_ki,
            pid_kd=self._cfgs.algo_cfgs.pid_kd,
            pid_d_delay=self._cfgs.algo_cfgs.pid_d_delay,
            pid_delta_p_ema_alpha=self._cfgs.algo_cfgs.pid_delta_p_ema_alpha,
            pid_delta_d_ema_alpha=self._cfgs.algo_cfgs.pid_delta_d_ema_alpha,
            sum_norm=self._cfgs.algo_cfgs.sum_norm,
            diff_norm=self._cfgs.algo_cfgs.diff_norm,
            penalty_max=self._cfgs.algo_cfgs.penalty_max,
            lagrangian_multiplier_init=self._cfgs.algo_cfgs.lagrangian_multiplier_init,
            cost_limit=self._cfgs.algo_cfgs.cost_limit,
        )

    def _update(self) -> None:
        super()._update()

        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]

        self._lagrange.pid_update(ep_cost)

        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier,
            },
        )