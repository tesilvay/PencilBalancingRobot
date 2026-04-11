from dataclasses import dataclass, field

import control as ct
import numpy as np

from src.shared import (
    ControlInput,
    PlantParams,
    State,
    TimingParams,
    WorkspaceParams,
    default_plant,
    default_timing,
    default_workspace,
    make_reference_state,
)

from .base import BaseController


@dataclass
class SmoothLQRParams:
    q_pos: float
    q_vel: float
    q_tilt: float
    q_tilt_rate: float
    q_u: float
    r_delta_u: float
    plant: PlantParams = field(default_factory=default_plant)
    timing: TimingParams = field(default_factory=default_timing)
    workspace: WorkspaceParams = field(default_factory=default_workspace)


SMOOTH_LQR_PRESETS = {
    "default": {
        "q_pos": 100.0,
        "q_vel": 0.001,
        "q_tilt": 10.0,
        "q_tilt_rate": 0.01,
        "q_u": 0.0,
        "r_delta_u": 5.0e5,
    },
    "smoother": {
        "q_pos": 10.0,
        "q_vel": 0.001,
        "q_tilt": 5.0,
        "q_tilt_rate": 0.01,
        "q_u": 0.0,
        "r_delta_u": 1.0e8,
    },
    "aggressive": {
        "base": "default",
        "q_pos": 5.0,
        "q_tilt": 30.0,
        "r_delta_u": 0.3,
    },
}


class SmoothLQRController(BaseController):
    """Discrete-time LQR on the augmented xi = [x; u_{k-1}] state with v = delta_u."""

    def __init__(self, params: SmoothLQRParams):
        from src.system.plant.dynamics_model import BuildLinearModel

        A_c, B_c = BuildLinearModel(params.plant)
        dt = float(params.timing.actuator_dt)
        x_ref = make_reference_state(params.workspace)

        n, m = A_c.shape[0], B_c.shape[1]
        sys_c = ct.ss(A_c, B_c, np.eye(n), np.zeros((n, m)))
        sys_d = ct.c2d(sys_c, dt)
        A_d = np.asarray(sys_d.A, dtype=float)
        B_d = np.asarray(sys_d.B, dtype=float)

        A_aug = np.block([
            [A_d, np.asarray(B_d, dtype=float)],
            [np.zeros((m, n)), np.eye(m)],
        ])
        B_aug = np.vstack([B_d, np.eye(m)])

        Q_axis = np.diag(
            [
                float(params.q_pos),
                float(params.q_vel),
                float(params.q_tilt),
                float(params.q_tilt_rate),
            ]
        )
        Q_x = np.block([
            [Q_axis, np.zeros_like(Q_axis)],
            [np.zeros_like(Q_axis), Q_axis],
        ])
        Q_u = float(params.q_u) * np.eye(m)
        Q = np.block([
            [Q_x, np.zeros((n, m))],
            [np.zeros((m, n)), Q_u],
        ])
        R = float(params.r_delta_u) * np.eye(m)

        self.K, _, _ = ct.dlqr(A_aug, B_aug, Q, R)
        self.K = np.asarray(self.K, dtype=float)

        self.x_ref = x_ref.as_vector()
        self.u_ref = (-np.linalg.pinv(B_c) @ (A_c @ self.x_ref)).ravel()
        self._u_prev = self.u_ref.copy()

    def compute(self, state: State) -> ControlInput:
        x = state.as_vector()
        x_err = x - self.x_ref
        u_err = self._u_prev - self.u_ref
        xi_err = np.concatenate([x_err, u_err])

        delta_u = -(self.K @ xi_err).ravel()
        u = self._u_prev + delta_u

        return ControlInput(float(u[0]), float(u[1]))

    def set_applied_command(self, u: ControlInput) -> None:
        self._u_prev = np.array([u.px_cmd, u.py_cmd], dtype=float)

    def reset(self, x_hat: State | None = None):
        if x_hat is None:
            self._u_prev = self.u_ref.copy()
            return

        self._u_prev = self.u_ref.copy()
        u = self.compute(x_hat)
        self._u_prev = np.array([u.px_cmd, u.py_cmd], dtype=float)
