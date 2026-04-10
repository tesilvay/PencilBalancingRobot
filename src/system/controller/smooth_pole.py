from dataclasses import dataclass, field

import numpy as np
import control as ct

from src.shared import (
    PlantParams,
    TimingParams,
    WorkspaceParams,
    State,
    ControlInput,
    default_plant,
    default_timing,
    default_workspace,
    make_reference_state,
)

from .base import BaseController


@dataclass
class SmoothPoleParams:
    max_pole:    float
    pole_step:    float
    slew_poles: float
    
    center_tilt_pos: float = 0.0   # rad per meter (or rad per mm if your state uses mm)
    center_tilt_vel: float = 0.0   # rad per (m/s)
    center_tilt_max: float = np.deg2rad(4.0)
    
    plant:      PlantParams     = field(default_factory=default_plant)
    timing:     TimingParams    = field(default_factory=default_timing)
    workspace:  WorkspaceParams = field(default_factory=default_workspace)


SMOOTH_POLE_PRESETS = {
    "default": {
        "max_pole": -12,
        "pole_step": 1,
        "slew_poles": 0.99,
        "center_tilt_pos": 0.0,      # example only if x is in meters
        "center_tilt_vel": 0.0,
        "center_tilt_max": np.deg2rad(3.0),
    },
    "lead": {
        "max_pole": -12,
        "pole_step": 1,
        "slew_poles": 0.98,
        "center_tilt_pos": 1e2*np.deg2rad(0.01),      # example only if x is in meters
        "center_tilt_vel": 0.0,
        "center_tilt_max": np.deg2rad(1.0),
    },
    "smoother":{
        "base": "default",
        "slew_poles": 0.99,
    },
    "test1": {
        "s_poles":    [-12,-13,-14,-15] * 2,
        "slew_poles": 0.99,
    },
    
}


class SmoothPolePlacementController(BaseController):
    """Discrete-time Δu feedback via augmented state ξ = [x; u_{k-1}], v = Δu."""

    def __init__(self, params: SmoothPoleParams):
        from src.system.plant.dynamics_model import BuildLinearModel

        self._params = params

        A_c, B_c = BuildLinearModel(params.plant)
        dt = params.timing.dt
        x_ref = make_reference_state(params.workspace)

        n, m = A_c.shape[0], B_c.shape[1]
        sys_c = ct.ss(A_c, B_c, np.eye(n), np.zeros((n, m)))
        sys_d = ct.c2d(sys_c, dt)
        A_d = np.array(sys_d.A)
        B_d = np.array(sys_d.B)

        poles_list = [
            params.max_pole - i * params.pole_step
            for i in range(n // 2)
        ] * 2

        z_from_s = np.exp(np.array(poles_list, dtype=complex) * dt)
        z_slew = np.full(m, params.slew_poles, dtype=complex)
        z = np.concatenate([z_from_s, z_slew])

        if z.size != n + m:
            raise ValueError(
                f"desired poles must have length {n + m} (dim ξ), got {z.size}"
            )

        A_aug = np.block([
            [A_d,                B_d],
            [np.zeros((m, n)),   np.eye(m)],
        ])
        B_aug = np.vstack([B_d, np.eye(m)])
        self.K = ct.place(A_aug, B_aug, z)

        self.x_ref_base = x_ref.as_vector()

        # Nominal equilibrium command only for the base reference.
        # Keep this fixed. Do NOT recompute it from the inward-lean reference,
        # because that inward lean is not a true steady-state equilibrium.
        self.u_ref = (-np.linalg.pinv(B_c) @ (A_c @ self.x_ref_base)).ravel()
        self._u_prev = self.u_ref.copy()

    def _soft_clip_angle(self, angle: float) -> float:
        max_angle = self._params.center_tilt_max
        if max_angle <= 0.0:
            return angle
        return max_angle * np.tanh(angle / max_angle)

    def _compute_centering_tilt_ref(self, pos_err: float, vel_err: float) -> float:
        raw_angle = (
            self._params.center_tilt_pos * pos_err
            + self._params.center_tilt_vel * vel_err
        )
        return -self._soft_clip_angle(raw_angle)

    def _build_state_error(self, x: np.ndarray) -> np.ndarray:
        e = x - self.x_ref_base

        # State order:
        # [x, x_dot, alpha_x, alpha_x_dot, y, y_dot, alpha_y, alpha_y_dot]

        alpha_x_ref = self._compute_centering_tilt_ref(
            pos_err=e[0],
            vel_err=e[1],
        )
        alpha_y_ref = self._compute_centering_tilt_ref(
            pos_err=e[4],
            vel_err=e[5],
        )

        e[2] = x[2] - alpha_x_ref
        e[6] = x[6] - alpha_y_ref

        return e

    def compute(self, state: State) -> ControlInput:
        x = state.as_vector()

        x_err = self._build_state_error(x)
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
    