from dataclasses import dataclass

import numpy as np
from src.shared import State, Measurement, ControlInput, NullParams

from .base import BaseEstimator

class FiniteDifferenceEstimator(BaseEstimator):

    def __init__(self, params: NullParams):
        super().__init__()
        self.prev_y_meas = None

    def estimate(
        self,
        y_meas: Measurement,
        dt: float,
        u_cmd: ControlInput | None = None,
    ) -> tuple[State, np.ndarray]:

        if self.prev_y_meas is None:
            vel = np.zeros(4)
        else:
            vel = np.array([
                (y_meas.px - self.prev_y_meas.px) / dt,
                (y_meas.ax - self.prev_y_meas.ax) / dt,
                (y_meas.py - self.prev_y_meas.py) / dt,
                (y_meas.ay - self.prev_y_meas.ay) / dt
            ])

        self.prev_y_meas = y_meas

        x_hat= State(
            px=y_meas.px,  vx=vel[0],
            ax=y_meas.ax,  wx=vel[1],
            py=y_meas.py,  vy=vel[2],
            ay=y_meas.ay,  wy=vel[3]
        )
        innovation = super().calc_innovation(y_meas, x_hat)
        return x_hat, innovation

    def reset(self):
        self.prev_y_meas = None
