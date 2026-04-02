from dataclasses import dataclass

import numpy as np
import control as ct
from core.sim_types import SystemState, PoseMeasurement, TableCommand

from perception.estimator_diagnostics import (
    build_kalman_snapshot,
)

from new_architecture.params import PlantParams, TimingParams

from .base import BaseEstimator
from .kalman_core import run_linear_kalman_step
from .lpf import LowPassFiniteDifferenceEstimator


@dataclass
class FullKalmanParams:
    plant: PlantParams
    timing: TimingParams
    q_pose_pos: float
    q_pose_ang: float
    q_vel_pos: float
    q_vel_ang: float
    r_pose_pos: float
    r_vel_pos: float
    r_pose_ang: float
    r_vel_ang: float
    lpf_alpha: float


FULL_KALMAN_PRESETS = {
    "default": {
        "plant": "default:default",
        "timing": "default:default",
        "q_pose_pos": 1e-8,
        "q_pose_ang": 1e-8,
        "q_vel_pos": 1e-4,
        "q_vel_ang": 1e-4,
        "r_pose_pos": 1e-7,
        "r_vel_pos": 1e-4,
        "r_pose_ang": 1e-4,
        "r_vel_ang": 1e-6,
        "lpf_alpha": 0.95,
    }
}


class FullStateKalmanFilter(BaseEstimator):
    """
    LPF finite-difference full state as measurement z ∈ R^8, fused with linear Kalman (H = I).
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        dt: float,
        Q: np.ndarray,
        R: np.ndarray,
        lpf: LowPassFiniteDifferenceEstimator,
    ):
        super().__init__()
        sys_c = ct.ss(A, B, np.eye(8), np.zeros((8, 2)))
        sys_d = ct.c2d(sys_c, dt)

        self.A = np.array(sys_d.A)
        self.B = np.array(sys_d.B)
        self.H = np.eye(8)
        self.Q = Q
        self.R = R
        self._lpf = lpf

        self.P = np.eye(8) * 0.01
        self.x_hat = np.zeros((8, 1))

    def update(
        self,
        pose: PoseMeasurement,
        dt: float,
        command_u: TableCommand | None = None,
    ) -> SystemState:
        z_state = self._lpf.update(pose, dt, command_u)
        z = z_state.as_vector().reshape(8, 1)

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
            alpha_y_dot=self.x_hat[7, 0],
        )

    def reset(self):
        super().reset()
        self._lpf.reset()
        self.P = np.eye(8) * 0.01
        self.x_hat = np.zeros((8, 1))
