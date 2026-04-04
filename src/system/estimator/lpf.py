from dataclasses import dataclass
import numpy as np

from src.shared import State, Measurement, ControlInput
from .base import BaseEstimator


@dataclass
class LPFParams:
    alpha_meas: float
    alpha_vel: float


LPF_PRESETS = {
    "default": {"alpha_meas": 0.95, "alpha_vel": 0.95}
}

class LowPassFiniteDifferenceEstimator(BaseEstimator):
    def __init__(self, params: LPFParams):
        super().__init__()
        self.alpha_meas = params.alpha_meas
        self.alpha_vel = params.alpha_vel

        self.prev_x_hat: State | None = None
        self.prev_y: Measurement | None = None
        self.prev_y_filt: np.ndarray | None = None  # fixed name
        self.prev_vel = np.zeros(4)

    def estimate(
        self, 
        y_meas: Measurement, 
        dt: float, 
        u_cmd: ControlInput
    ) -> tuple[State, np.ndarray]:
        
        y_vec = y_meas.as_vector()

        if self.prev_y is None:
            y_filt = y_vec.copy()
            vel = np.zeros(4)
        else:
            y_filt = self.alpha_meas * self.prev_y_filt + (1 - self.alpha_meas) * y_vec  # fixed
            raw_vel = (y_filt - self.prev_y_filt) / dt                                  # fixed
            vel = self.alpha_vel * self.prev_vel + (1 - self.alpha_vel) * raw_vel

        x_hat = State(
            px=y_filt[0], vx=vel[0],
            ax=y_filt[1], wx=vel[1],
            py=y_filt[2], vy=vel[2],
            ay=y_filt[3], wy=vel[3],
        )

        self.prev_y = y
        self.prev_y_filt = y_filt  # fixed
        self.prev_vel = vel
        self.prev_x_hat = x_hat
        innovation = super().calc_innovation(y_meas, x_hat)
        return x_hat, innovation

    def reset(self):
        self.prev_x_hat = None
        self.prev_y = None
        self.prev_y_filt = None  # fixed
        self.prev_vel = np.zeros(4)