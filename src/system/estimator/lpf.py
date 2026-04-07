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
class LPFParams:
    alpha_meas: float
    alpha_vel: float
    plant: PlantParams = field(default_factory=default_plant)
    timing: TimingParams = field(default_factory=default_timing)


LPF_PRESETS = {
    "default": {"alpha_meas": 0.0, "alpha_vel": 0.95},
    "test": {"alpha_meas": 0.75, "alpha_vel": 0.75},
    "smoother": {"alpha_meas": 0.8, "alpha_vel": 0.8},
    "test2": {"alpha_meas": 0.8, "alpha_vel": 0.8},
}


class LowPassFiniteDifferenceEstimator(BaseEstimator):
    def __init__(self, params: LPFParams):
        super().__init__()
        self.alpha_meas = params.alpha_meas
        self.alpha_vel = params.alpha_vel
        self.A, self.B = discretize_AB(params.plant, params.timing)
        self.H = measurement_H()

        self._x_post = np.zeros((8, 1))
        self.prev_y: Measurement | None = None
        self.prev_y_filt: np.ndarray | None = None
        self.prev_vel = np.zeros(4)

    def estimate(
        self,
        y_meas: Measurement,
        dt: float,
        u_cmd: ControlInput | None,
    ) -> tuple[State, np.ndarray]:

        z = self.measurement_z(y_meas)
        u = self.control_u(u_cmd, y_meas)
        innovation, _ = self._step_prediction(z, self._x_post, u)

        y_vec = y_meas.as_vector()

        if self.prev_y is None:
            y_filt = y_vec.copy()
            vel = np.zeros(4)
        else:
            y_filt = self.alpha_meas * self.prev_y_filt + (1 - self.alpha_meas) * y_vec
            raw_vel = (y_filt - self.prev_y_filt) / dt
            vel = self.alpha_vel * self.prev_vel + (1 - self.alpha_vel) * raw_vel

        x_hat = State(
            px=y_filt[0], vx=vel[0],
            ax=y_filt[1], wx=vel[1],
            py=y_filt[2], vy=vel[2],
            ay=y_filt[3], wy=vel[3],
        )

        self.prev_y = y_meas
        self.prev_y_filt = y_filt
        self.prev_vel = vel
        self._x_post = x_hat.as_vector().reshape(-1, 1)

        return x_hat, innovation

    def reset(self, x_hat: State | None = None):
        if x_hat is None:
            self._x_post = np.zeros((8, 1))
            self.prev_y = None
            self.prev_y_filt = None
            self.prev_vel = np.zeros(4)
        else:
            self._x_post = x_hat.as_vector().reshape(-1, 1)
            self.prev_y = None
            self.prev_y_filt = np.array([x_hat.px, x_hat.ax, x_hat.py, x_hat.ay])
            self.prev_vel = np.array([x_hat.vx, x_hat.wx, x_hat.vy, x_hat.wy])
