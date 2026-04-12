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
    max_delta_u: float | None = None
    catch_max_delta_u: float | None = None
    catch_angle_deg: float = 3.0
    catch_projection_gain: float = 0.0
    catch_velocity_lookahead_s: float = 0.0
    max_command_radius: float | None = None
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
    "real": {
        "q_pos": 2.5,
        "q_vel": 0.001,
        "q_tilt": 10.0,
        "q_tilt_rate": 0.001,
        "q_u": 0.1,
        "r_delta_u": 1.0e6,
        "max_delta_u": 2.5e-3,
        
        "catch_max_delta_u": 15.0e-3,
        "catch_angle_deg": 3.0,
        "catch_projection_gain": 1.15,
        
        "catch_velocity_lookahead_s": 0.06,
        
        "max_command_radius": 7.0e-2,
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
        self.max_delta_u = params.max_delta_u
        self.catch_max_delta_u = params.catch_max_delta_u
        self.catch_angle_rad = float(np.deg2rad(params.catch_angle_deg))
        self.catch_projection_gain = float(params.catch_projection_gain)
        self.catch_velocity_lookahead_s = float(params.catch_velocity_lookahead_s)
        self.com_length = float(params.plant.com_length)
        self.max_command_radius = params.max_command_radius
        self.workspace_ref = np.array(
            [float(params.workspace.x_ref), float(params.workspace.y_ref)],
            dtype=float,
        )

    def _catch_strength(self, x: np.ndarray) -> float:
        angle_norm = float(np.linalg.norm([x[2], x[6]]))
        if self.catch_angle_rad <= 0.0:
            return 1.0
        return float(np.clip(angle_norm / self.catch_angle_rad, 0.0, 1.0))

    def _effective_x_ref(self, x: np.ndarray) -> np.ndarray:
        x_ref = self.x_ref.copy()
        if self.catch_projection_gain == 0.0 and self.catch_velocity_lookahead_s == 0.0:
            return x_ref

        ax = float(x[2])
        wx = float(x[3])
        ay = float(x[6])
        wy = float(x[7])
        catch_offset = np.array(
            [
                self.catch_projection_gain * self.com_length * np.sin(ax)
                + self.catch_velocity_lookahead_s * self.com_length * np.cos(ax) * wx,
                self.catch_projection_gain * self.com_length * np.sin(ay)
                + self.catch_velocity_lookahead_s * self.com_length * np.cos(ay) * wy,
            ],
            dtype=float,
        )
        strength = self._catch_strength(x)
        x_ref[0] = self.workspace_ref[0] + strength * catch_offset[0]
        x_ref[4] = self.workspace_ref[1] + strength * catch_offset[1]
        return x_ref

    def _limit_delta_u(self, delta_u: np.ndarray, x: np.ndarray) -> np.ndarray:
        if self.max_delta_u is None and self.catch_max_delta_u is None:
            return delta_u

        base_max_delta = np.inf if self.max_delta_u is None else float(self.max_delta_u)
        catch_max_delta = base_max_delta if self.catch_max_delta_u is None else float(self.catch_max_delta_u)
        strength = self._catch_strength(x)
        max_delta = (1.0 - strength) * base_max_delta + strength * catch_max_delta
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
        x = state.as_vector()
        x_err = x - self._effective_x_ref(x)
        u_err = self._u_prev - self.u_ref
        xi_err = np.concatenate([x_err, u_err])

        delta_u = -(self.K @ xi_err).ravel()
        delta_u = self._limit_delta_u(delta_u, x)
        u = self._u_prev + delta_u
        u = self._limit_command_radius(u)

        return ControlInput(float(u[0]), float(u[1]))

    def set_applied_command(self, u: ControlInput, state: State) -> None:
        del state
        self._u_prev = np.array([u.px_cmd, u.py_cmd], dtype=float)

    def reset(self, x_hat: State | None = None):
        if x_hat is None:
            self._u_prev = self.u_ref.copy()
            return

        self._u_prev = self.u_ref.copy()
        u = self.compute(x_hat)
        self._u_prev = np.array([u.px_cmd, u.py_cmd], dtype=float)
