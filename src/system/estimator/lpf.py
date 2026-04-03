from dataclasses import dataclass

import numpy as np
from src.shared import SystemState, PoseMeasurement, TableCommand

from .base import BaseEstimator


@dataclass
class LPFParams:
    alpha: float


LPF_PRESETS = {"default": {"alpha": 0.93}}


class LowPassFiniteDifferenceEstimator(BaseEstimator):

    def __init__(self, params: LPFParams):
        super().__init__()
        self.prev_pose = None
        self.prev_vel = np.zeros(4)
        self.alpha = params.alpha

    def update(
        self,
        pose: PoseMeasurement,
        dt: float,
        command_u: TableCommand | None = None,
    ) -> SystemState:

        if self.prev_pose is None:
            vel = np.zeros(4)
        else:
            raw_vel = np.array([
                (pose.X - self.prev_pose.X) / dt,
                (pose.alpha_x - self.prev_pose.alpha_x) / dt,
                (pose.Y - self.prev_pose.Y) / dt,
                (pose.alpha_y - self.prev_pose.alpha_y) / dt
            ])

            vel = self.alpha * self.prev_vel + (1 - self.alpha) * raw_vel

        self.prev_pose = pose
        self.prev_vel = vel

        return SystemState(
            px=pose.X,
            vx=vel[0],
            ax=pose.alpha_x,
            wx=vel[1],
            py=pose.Y,
            vy=vel[2],
            ay=pose.alpha_y,
            wy=vel[3]
        )

    def reset(self):
        super().reset()
        self.prev_pose = None
        self.prev_vel = np.zeros(4)
