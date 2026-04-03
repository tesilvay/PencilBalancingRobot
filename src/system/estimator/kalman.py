from dataclasses import dataclass

import numpy as np
import control as ct
from src.shared import SystemState, PoseMeasurement, TableCommand

from perception.estimator_diagnostics import (
    build_kalman_snapshot,
)

from new_architecture.params import PlantParams, TimingParams

from .base import BaseEstimator
from .kalman_core import run_linear_kalman_step


@dataclass
class KalmanParams:
    plant: PlantParams
    timing: TimingParams
    q_pose_pos: float
    q_pose_ang: float
    q_vel_pos: float
    q_vel_ang: float
    r_pose_pos: float
    r_pose_ang: float


KALMAN_PRESETS = {
    "default": {
        "plant": "default:default",
        "timing": "default:default",
        "q_pose_pos": 1e-6,
        "q_pose_ang": 1e-6,
        "q_vel_pos": 1e-3,
        "q_vel_ang": 1e-2,
        "r_pose_pos": 1e-2,
        "r_pose_ang": 7e-2,
    }
}


class KalmanEstimator(BaseEstimator):

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        dt: float,
        Q: np.ndarray,
        R: np.ndarray,
    ):
        super().__init__()
        # Discretize continuous system with control input u = [x_des, y_des]
        sys_c = ct.ss(A, B, np.eye(8), np.zeros((8, 2)))
        sys_d = ct.c2d(sys_c, dt)

        self.A = np.array(sys_d.A)
        self.B = np.array(sys_d.B)

        # Measurement matrix
        # z = [X, alpha_x, Y, alpha_y]
        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1.0  # X
        self.H[1, 2] = 1.0  # alpha_x
        self.H[2, 4] = 1.0  # Y
        self.H[3, 6] = 1.0  # alpha_y

        self.Q = Q
        self.R = R

        self.P_init = np.eye(8) * 2e-2 # bigger makes search more
        self.x_hat_init = np.zeros((8, 1))

        self.P = self.P_init
        #self.P = solve_discrete_are(A.T, self.H.T, self.Q, self.R)
        self.x_hat = self.x_hat_init

    def update(
        self,
        pose: PoseMeasurement,
        dt: float,
        command_u: TableCommand | None = None,
    ) -> SystemState:

        z = np.array(
            [pose.X, pose.alpha_x, pose.Y, pose.alpha_y], dtype=float
        ).reshape(-1, 1)

        if command_u is None:
            u = np.zeros((2, 1))
        else:
            u = np.array([[command_u.x_des], [command_u.y_des]])

        step = run_linear_kalman_step(
            self.A,
            self.B,
            self.H,
            self.Q,
            self.R,
            self.x_hat,
            self.P,
            z,
            u,
        )
        self.x_hat = step.x_hat
        self.P = step.P
        self._last_diagnostic_snapshot = build_kalman_snapshot(
            estimator_name=type(self).__name__,
            step_idx=self._diag_step_idx,
            t_s=self._diag_t_s,
            dt_s=dt,
            measurement_fresh=self._diag_measurement_fresh,
            z_changed=self._diag_z_changed,
            step=step,
        )

        return SystemState(
            x=self.x_hat[0, 0],
            x_dot=self.x_hat[1, 0],
            alpha_x=self.x_hat[2, 0],
            alpha_x_dot=self.x_hat[3, 0],
            y=self.x_hat[4, 0],
            y_dot=self.x_hat[5, 0],
            alpha_y=self.x_hat[6, 0],
            alpha_y_dot=self.x_hat[7, 0]
        )

    def reset(self):
        super().reset()
        self.P = self.P_init
        self.x_hat = self.x_hat_init
