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
class KalmanParams:
    q_y_meas_pos: float
    q_y_meas_ang: float
    q_vel_pos: float
    q_vel_ang: float
    r_y_meas_pos: float
    r_y_meas_ang: float
    plant: PlantParams = field(default_factory=default_plant)
    timing: TimingParams = field(default_factory=default_timing)


KALMAN_PRESETS = {
    "default": {
        "q_y_meas_pos": 1e-6,
        "q_y_meas_ang": 1e-6,
        "q_vel_pos": 1e-3,
        "q_vel_ang": 1e-2,
        "r_y_meas_pos": 1e-2,
        "r_y_meas_ang": 7e-2,
    },
    "test":{
        "q_y_meas_pos": 1e-4,
        "q_y_meas_ang": 1e-4,
        "q_vel_pos": 1e-2,
        "q_vel_ang": 1e-2,
        "r_y_meas_pos": 1e-3,
        "r_y_meas_ang": 1e-3,
    },
    "test1":{
        "q_y_meas_pos": 1e-4,
        "q_y_meas_ang": 1e-4,
        "q_vel_pos": 1e-1,
        "q_vel_ang": 1e-1,
        "r_y_meas_pos": 1e-3,
        "r_y_meas_ang": 1e-3,
    }
}


class KalmanEstimator(BaseEstimator):

    def __init__(self, params: KalmanParams):
        super().__init__()
        self.A, self.B = discretize_AB(params.plant, params.timing)
        self.H = measurement_H()

        p = params
        self.Q = np.diag([
            p.q_y_meas_pos, p.q_vel_pos, p.q_y_meas_ang, p.q_vel_ang,
            p.q_y_meas_pos, p.q_vel_pos, p.q_y_meas_ang, p.q_vel_ang,
        ])
        self.R = np.diag([
            p.r_y_meas_pos, p.r_y_meas_ang,
            p.r_y_meas_pos, p.r_y_meas_ang,
        ])

        self.P_init = np.eye(8) * 2e-1
        self.x_hat_init_0 = np.zeros((8, 1))
        self.P = self.P_init.copy()
        self.x_hat = self.x_hat_init_0.copy()

    def estimate(
        self,
        y_meas: Measurement,
        dt: float,
        u_cmd: ControlInput | None,
    ) -> tuple[State, np.ndarray]:

        z = self.measurement_z(y_meas)
        u = self.control_u(u_cmd, y_meas)

        innovation, x_pred = self._step_prediction(z, self.x_hat, u)
        P_pred = self.A @ self.P @ self.A.T + self.Q

        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x_hat = x_pred + K @ innovation.reshape(-1, 1)
        self.P = (np.eye(8) - K @ self.H) @ P_pred

        x_hat = State.from_iterable(self.x_hat.flatten())

        return x_hat, innovation

    def reset(self, x_hat: State | None = None):
        self.P = self.P_init.copy()

        if x_hat:
            self.x_hat = x_hat.as_vector().reshape(-1, 1)
        else:
            self.x_hat = self.x_hat_init_0.copy()