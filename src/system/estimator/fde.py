from dataclasses import dataclass, field

import numpy as np

from src.shared import (
    State,
    Measurement,
    ControlInput,
    PlantParams,
    TimingParams,
    default_plant,
    default_timing,
)

from .base import BaseEstimator
from .dynamics_disc import discretize_AB, measurement_H


@dataclass
class FdeParams:
    plant: PlantParams = field(default_factory=default_plant)
    timing: TimingParams = field(default_factory=default_timing)


FDE_PRESETS = {"default": {}}


class FiniteDifferenceEstimator(BaseEstimator):

    def __init__(self, params: FdeParams):
        super().__init__()
        self.A, self.B = discretize_AB(params.plant, params.timing)
        self.H = measurement_H()

        self._x_post = np.zeros((8, 1))
        self.prev_y_meas: Measurement | None = None

    def estimate(
        self,
        y_meas: Measurement,
        dt: float,
        u_cmd: ControlInput | None = None,
    ) -> tuple[State, np.ndarray]:

        z = self.measurement_z(y_meas)
        u = self.control_u(u_cmd, y_meas)
        innovation, _ = self._step_prediction(z, self._x_post, u)

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

        x_hat = State(
            px=y_meas.px, vx=vel[0],
            ax=y_meas.ax, wx=vel[1],
            py=y_meas.py, vy=vel[2],
            ay=y_meas.ay, wy=vel[3]
        )
        self._x_post = x_hat.as_vector().reshape(-1, 1)

        return x_hat, innovation

    def reset(self, x_hat: State | None = None):
        self.prev_y_meas = None
        if x_hat is not None:
            self._x_post = x_hat.as_vector().reshape(-1, 1)
        else:
            self._x_post = np.zeros((8, 1))
