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
class DeltaLQRParams:
    pos_scale: float
    vel_scale: float
    tilt_scale: float
    tilt_rate_scale: float
    
    q_pos: float # cares about bringing table to center
    q_vel: float # tries to eliminate drift
    q_tilt: float # tries to set pencil upright
    q_tilt_rate: float # penalty on angular acceleration
    
    q_command: float # too little q_command: command may wander too freely ?????
    
    r_delta_u: float # penalty on control command
    
    max_delta_u: float | None = None
    max_command_radius: float | None = None
    plant: PlantParams = field(default_factory=default_plant)
    timing: TimingParams = field(default_factory=default_timing)
    workspace: WorkspaceParams = field(default_factory=default_workspace)


DELTA_LQR_PRESETS = {
    "default": {
        
        # “scale of concern”, not “scale of catastrophe”
        # at what scale do they matter?
        # we care about pos when we are about 2cm away right?
        # or about tilt when at more than 2 degrees
        "pos_scale": 2e-2,
        "vel_scale": 5e-2, # /s
        "tilt_scale": np.deg2rad(2.0),
        "tilt_rate_scale": np.deg2rad(10), # /s
        
        "q_pos": 0.5,
        "q_vel": 1.0e-8,
        "q_tilt": 0.001,
        "q_tilt_rate": 1.0e-8,
        
        "q_command": 0.05,
        
        "r_delta_u": 6.0e8,
        
        "max_delta_u": 4.0e-2,
        "max_command_radius": 8.0e-2
    },
    "gentle": {
        "base": "default",
        "q_pos": 5.0,
        "q_tilt": 15.0,
        "r_delta_u": 5.0e5,
        "max_delta_u": 5.0e-4,
    },
    "stronger": {
        "base": "default",
        "q_pos": 25.0,
        "q_tilt": 50.0,
        "r_delta_u": 2.5e4,
        "max_delta_u": 2.0e-3,
    },
}


class DeltaLQRController(BaseController):
    """Discrete LQR that optimizes command increments and returns absolute position."""

    def __init__(self, params: DeltaLQRParams):
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
            [A_d, B_d],
            [np.zeros((m, n)), np.eye(m)],
        ])
        B_aug = np.vstack([B_d, np.eye(m)])

        
        # Normalized Q
        Q_axis = np.diag([
            params.q_pos / params.pos_scale**2,
            params.q_vel / params.vel_scale**2,
            params.q_tilt / params.tilt_scale**2,
            params.q_tilt_rate / params.tilt_rate_scale**2,
        ])
        
        Q_x = np.block([
            [Q_axis, np.zeros_like(Q_axis)],
            [np.zeros_like(Q_axis), Q_axis],
        ])
        Q_command = float(params.q_command) * np.eye(m)
        Q = np.block([
            [Q_x, np.zeros((n, m))],
            [np.zeros((m, n)), Q_command],
        ])
        R = float(params.r_delta_u) * np.eye(m)

        self.K, _, _ = ct.dlqr(A_aug, B_aug, Q, R)
        self.K = np.asarray(self.K, dtype=float)

        self.x_ref = x_ref.as_vector()
        self.u_ref = (-np.linalg.pinv(B_c) @ (A_c @ self.x_ref)).ravel()
        self._u_prev = self.u_ref.copy()
        self.max_delta_u = params.max_delta_u
        self.max_command_radius = params.max_command_radius
        self.workspace_ref = np.array(
            [float(params.workspace.x_ref), float(params.workspace.y_ref)],
            dtype=float,
        )

    def _limit_delta_u(self, delta_u: np.ndarray) -> np.ndarray:
        if self.max_delta_u is None:
            return delta_u

        max_delta = float(self.max_delta_u)
        if max_delta <= 0.0:
            return np.zeros_like(delta_u)

        delta_norm = float(np.linalg.norm(delta_u))
        if delta_norm <= max_delta or delta_norm <= 0.0:
            return delta_u

        return delta_u * (max_delta / delta_norm)

    def _limit_command_radius(self, u: np.ndarray) -> np.ndarray:
        if self.max_command_radius is None:
            return u

        max_radius = float(self.max_command_radius)
        if max_radius <= 0.0:
            return self.workspace_ref.copy()

        radial = u - self.workspace_ref
        radius = float(np.linalg.norm(radial))
        if radius <= max_radius or radius <= 0.0:
            return u

        return self.workspace_ref + radial * (max_radius / radius)

    def compute(self, state: State) -> ControlInput:
        x_err = state.as_vector() - self.x_ref
        u_err = self._u_prev - self.u_ref
        xi_err = np.concatenate([x_err, u_err])

        delta_u = -(self.K @ xi_err).ravel()
        delta_u = self._limit_delta_u(delta_u)
        u = self._u_prev + delta_u
        u = self._limit_command_radius(u)

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
