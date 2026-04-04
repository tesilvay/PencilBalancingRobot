from dataclasses import dataclass, field

import numpy as np
import control as ct
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
from .kalman_core import run_linear_kalman_step


@dataclass
class KalmanParams:
    q_y_meas_pos: float
    q_y_meas_ang: float
    q_vel_pos:  float
    q_vel_ang:  float
    r_y_meas_pos: float
    r_y_meas_ang: float
    plant:      PlantParams  = field(default_factory=default_plant)
    timing:     TimingParams = field(default_factory=default_timing)


KALMAN_PRESETS = {
    "default": {
        "q_y_meas_pos": 1e-6,
        "q_y_meas_ang": 1e-6,
        "q_vel_pos":  1e-3,
        "q_vel_ang":  1e-2,
        "r_y_meas_pos": 1e-2,
        "r_y_meas_ang": 7e-2,
    }
}


class KalmanEstimator(BaseEstimator):

    def __init__(self, params: KalmanParams):
        super().__init__()
        from src.system.plant.dynamics_model import BuildLinearModel
        A_c, B_c = BuildLinearModel(params.plant)

        sys_c = ct.ss(A_c, B_c, np.eye(8), np.zeros((8, 2)))
        sys_d = ct.c2d(sys_c, params.timing.dt)
        self.A = np.array(sys_d.A)
        self.B = np.array(sys_d.B)

        # z = [X, alpha_x, Y, alpha_y]
        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1.0
        self.H[1, 2] = 1.0
        self.H[2, 4] = 1.0
        self.H[3, 6] = 1.0

        p = params
        self.Q = np.diag([
            p.q_y_meas_pos, p.q_vel_pos, p.q_y_meas_ang, p.q_vel_ang,
            p.q_y_meas_pos, p.q_vel_pos, p.q_y_meas_ang, p.q_vel_ang,
        ])
        self.R = np.diag([
            p.r_y_meas_pos, p.r_y_meas_ang,
            p.r_y_meas_pos, p.r_y_meas_ang,
        ])

        self.P_init       = np.eye(8) * 2e-2
        self.x_hat_init_0 = np.zeros((8, 1))
        self.P            = self.P_init.copy()
        self.x_hat        = self.x_hat_init_0.copy()

    def estimate(
        self, 
        y_meas: Measurement, 
        dt: float, 
        u_cmd: ControlInput
    ) -> tuple[State, np.ndarray]:

        z = np.array(
            [y_meas.px, y_meas.ax, y_meas.py, y_meas.ay], dtype=float
        ).reshape(-1, 1)     

        if u_cmd is None:
            # if no command, it's because the table isn't moving
            # it means it's just below the pencil, nothing else
            u = np.array([[y_meas.px], [y_meas.py]])
        else:
            u = np.array([[u_cmd.px_cmd], [u_cmd.py_cmd]])

        # ----- Prediction -----
        x_pred = self.A @ self.x_hat + self.B @ u
        P_pred = self.A @ self.P @ self.A.T + self.Q

        # ----- Update -----
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        innovation = z - self.H @ x_pred

        self.x_hat = x_pred + K @ innovation
        self.P = (np.eye(8) - K @ self.H) @ P_pred
        
        x_hat = State.from_iterable(self.x_hat.flatten())

        return x_hat, innovation
        
    def reset(self, x_hat : State | None = None):
        self.P = self.P_init.copy()
        
        if x_hat:
            self.x_hat = x_hat.as_vector()
        else:        
            self.x_hat = self.x_hat_init_0.copy()
