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
    delta_u_scale: float

    q_pos: float # cares about bringing table to center
    q_vel: float # tries to eliminate drift
    q_tilt: float # tries to set pencil upright
    q_tilt_rate: float # penalty on angular acceleration


    # penalizing u_err from u_ref_lqr, which itself is moving.
    # This is fine and actually helpful
    # it stops u_prev from wandering away from the shifted reference.
    q_command: float

    r_delta_u: float # penalty on control command


    pos_ref_ki: float
    max_pos_ref_shift: float

    tilt_ref_ki: float
    max_tilt_ref_shift: float

    tilt_calib_ki: float
    max_tilt_calib: float
    min_pos_activate_calib: float

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
        "pos_scale": 1e-2,
        "vel_scale": 3e-2, # /s
        "tilt_scale": np.deg2rad(2.0),
        "tilt_rate_scale": np.deg2rad(10), # /s
        "delta_u_scale": 1e-3,

        "q_pos": 0.02,
        "q_vel": 0.015,
        "q_tilt": 1.0,
        "q_tilt_rate": 0.01,

        "q_command": 0.06,

        "r_delta_u": 5.0,

        "max_delta_u": 2.0e-3,
        "max_command_radius": 7.0e-2,


        "pos_ref_ki": 0.0,
        "max_pos_ref_shift": 50e-2,

        "tilt_ref_ki": 0.8,
        "max_tilt_ref_shift": np.deg2rad(4.0),

        "tilt_calib_ki": 0.8,
        "max_tilt_calib": np.deg2rad(0.3),
        "min_pos_activate_calib": 2.0e-2,

    },
    "gentle": {
        "base": "default",

        "r_delta_u": 1050,
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
        R = (params.r_delta_u / params.delta_u_scale**2) * np.eye(m)

        self.K, _, _ = ct.dlqr(A_aug, B_aug, Q, R)
        self.K = np.asarray(self.K, dtype=float)


        self._x_ref_true = x_ref.as_vector()  # never changes
        self.x_ref_lqr = x_ref.as_vector().copy()  # drifts with integrator
        self._u_ref_true = -np.linalg.pinv(B_c) @ A_c  # [m x n], computed once
        self.u_ref_lqr = (self._u_ref_true @ self.x_ref_lqr).ravel()

        self._u_prev = self.u_ref_lqr.copy()
        self.max_delta_u = params.max_delta_u
        self.max_command_radius = params.max_command_radius
        self.workspace_ref = np.array(
            [float(params.workspace.x_ref), float(params.workspace.y_ref)],
            dtype=float,
        )

        # Integrator: shifting reference
        self.pos_ref_ki = params.pos_ref_ki
        self.max_pos_ref_shift = params.max_pos_ref_shift

        self.tilt_ref_ki = float(params.tilt_ref_ki)
        self.max_tilt_ref_shift = float(params.max_tilt_ref_shift)

        self.tilt_calib_ki = float(params.tilt_calib_ki)
        self.max_tilt_calib = float(params.max_tilt_calib)
        self.min_pos_activate_calib = float(params.min_pos_activate_calib)
        self.calib_dwell_time = 0.4  # seconds
        self._calib_far_timer = 0.0
        self._calib_active = False

        self.tilt_calib_x = 0
        self.tilt_calib_y = 0

        self.actuator_dt = float(params.timing.actuator_dt)

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

    def _print_ref(self) -> None:
        print(f"ref: px: {self.x_ref_lqr[0]*1000:.2f} py: {self.x_ref_lqr[4]*1000:.2f} ax: {np.rad2deg(self.x_ref_lqr[2]):.3f} ay: {np.rad2deg(self.x_ref_lqr[6]):.3f}")

    def reference_state(self) -> State:
        return State.from_iterable(self.x_ref_lqr)

    def _refresh_u_ref(self) -> None:
        self.u_ref_lqr = (self._u_ref_true @ self.x_ref_lqr).ravel()
        
    def _clamp(self, value: float, max_value: float, init_val:float) ->  float:
        clamped_value = np.clip(
            value,
            init_val - max_value,
            init_val + max_value,
        )
        return clamped_value

    def _update_integrator(self, state: State) -> None:
        # Position
        pos_err_x = state.px - self._x_ref_true[0]
        pos_err_y = state.py - self._x_ref_true[4]
        self.x_ref_lqr[0] -= self.pos_ref_ki * pos_err_x * self.actuator_dt
        self.x_ref_lqr[4] -= self.pos_ref_ki * pos_err_y * self.actuator_dt
        
        # clamp
        self.x_ref_lqr[0] = self._clamp(self.x_ref_lqr[0], self.max_pos_ref_shift, self._x_ref_true[0])
        self.x_ref_lqr[4] = self._clamp(self.x_ref_lqr[4], self.max_pos_ref_shift, self._x_ref_true[4])

        # Tilt — use bias-corrected measurement
        tilt_err_x = (state.ax + self.tilt_calib_x) - self._x_ref_true[2]
        tilt_err_y = (state.ay + self.tilt_calib_y) - self._x_ref_true[6]
        self.x_ref_lqr[2] -= self.tilt_ref_ki * tilt_err_x * self.actuator_dt
        self.x_ref_lqr[6] -= self.tilt_ref_ki * tilt_err_y * self.actuator_dt
        
        # clamp
        self.x_ref_lqr[2] = self._clamp(self.x_ref_lqr[2], self.max_tilt_ref_shift, self._x_ref_true[2])
        self.x_ref_lqr[6] = self._clamp(self.x_ref_lqr[6], self.max_tilt_ref_shift, self._x_ref_true[6])



        # positive because the positive x error, means we are tilted too positively
        # if we add tilt bias positively, we correct the bias saying we are "upright" when we aren't
        # the bias is essentially left, so we need to add right
        pos_err = np.array([pos_err_x, pos_err_y])
        pos_norm = np.linalg.norm(pos_err)
        
        activate_threshold = self.min_pos_activate_calib
        deactivate_threshold = 0.8 * self.min_pos_activate_calib
        
        if self._calib_active:
            # stay active until clearly back near center
            if pos_norm <= deactivate_threshold:
                self._calib_active = False
                self._calib_far_timer = 0.0
        else:
            # only activate if far for long enough
            if pos_norm >= activate_threshold:
                self._calib_far_timer += self.actuator_dt
                if self._calib_far_timer >= self.calib_dwell_time:
                    self._calib_active = True
            else:
                self._calib_far_timer = 0.0
        
        if self._calib_active:
            self.tilt_calib_x += self.tilt_calib_ki * pos_err_x * self.actuator_dt
            self.tilt_calib_y += self.tilt_calib_ki * pos_err_y * self.actuator_dt

        # clamp
        self.tilt_calib_x = self._clamp(self.tilt_calib_x, self.max_tilt_calib, init_val=0)
        self.tilt_calib_y = self._clamp(self.tilt_calib_y, self.max_tilt_calib, init_val=0)

        self._refresh_u_ref()
        #self._print_ref()

    def compute(self, state: State) -> ControlInput:

        x_err = state.as_vector() - self.x_ref_lqr

        # remove estimated sensor bias
        x_err[2] += self.tilt_calib_x
        x_err[6] += self.tilt_calib_y

        u_err = self._u_prev - self.u_ref_lqr
        xi_err = np.concatenate([x_err, u_err])

        delta_u = -(self.K @ xi_err).ravel()
        delta_u = self._limit_delta_u(delta_u)
        u = self._u_prev + delta_u
        u = self._limit_command_radius(u)

        return ControlInput(float(u[0]), float(u[1]))

    def set_applied_command(self, u: ControlInput, state: State) -> None:
        self._u_prev = np.array([u.px_cmd, u.py_cmd], dtype=float)
        self._update_integrator(state)

    def reset(self, x_hat: State | None = None):
        print("delta_lqr reset", x_hat)

        # reset integrator
        self.x_ref_lqr = self._x_ref_true.copy()
        self.u_ref_lqr = (self._u_ref_true @ self.x_ref_lqr).ravel()
        self._integrator_active = False

        self.tilt_calib_x = 0
        self.tilt_calib_y = 0
        
        self._calib_far_timer = 0.0
        self._calib_active = False

        self._u_prev = self.u_ref_lqr.copy()
        if x_hat is not None:
            u = self.compute(x_hat)
            self._u_prev = np.array([u.px_cmd, u.py_cmd], dtype=float)


