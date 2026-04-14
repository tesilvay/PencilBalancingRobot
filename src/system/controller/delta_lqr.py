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
    
    pos_thresh: float
    angle_thresh: float
    rate_thresh: float | None
    ki: float
    
    tilt_stale_time_s: float
    tilt_deadband: float
    tilt_ki: float
    max_tilt_bias: float
    
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
        
        "q_pos": 0.05,
        "q_vel": 0,
        "q_tilt": 1.0,
        "q_tilt_rate": 0,
        
        "q_command": 0.06,
        
        "r_delta_u": 5.0,
        
        "max_delta_u": 8.0e-3,
        "max_command_radius": 7.0e-2,
        
        "pos_thresh": 1.5e-2,
        "angle_thresh": np.deg2rad(3.6),
        "rate_thresh": None,
        "ki": 0.1,
        
        "tilt_stale_time_s": 0.4,
        "tilt_deadband": np.deg2rad(0.1),
        "tilt_ki": 0.1,
        "max_tilt_bias": np.deg2rad(2.0),
        
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
        self._x_ref_lqr = x_ref.as_vector().copy()  # drifts with integrator
        self._u_ref_true = -np.linalg.pinv(B_c) @ A_c  # [m x n], computed once
        self.u_ref_lqr = (self._u_ref_true @ self._x_ref_lqr).ravel()
        
        self._u_prev = self.u_ref_lqr.copy()
        self.max_delta_u = params.max_delta_u
        self.max_command_radius = params.max_command_radius
        self.workspace_ref = np.array(
            [float(params.workspace.x_ref), float(params.workspace.y_ref)],
            dtype=float,
        )
        
        # Integrator: shifting reference
        self._integrator_active = False
        self._tilt_x_integrator_active = False
        self._tilt_y_integrator_active = False
        self._tilt_x_same_sign_time = 0.0
        self._tilt_y_same_sign_time = 0.0
        self._tilt_x_prev_sign = 0.0
        self._tilt_y_prev_sign = 0.0
        self.pos_thresh = params.pos_thresh
        self.angle_thresh = params.angle_thresh
        self.rate_thresh = params.rate_thresh
        self.ki = params.ki
        self.tilt_stale_time_s = float(params.tilt_stale_time_s)
        self.tilt_deadband = float(params.tilt_deadband)
        self.tilt_ki = float(params.tilt_ki)
        self.max_tilt_bias = float(params.max_tilt_bias)
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
        print(f"ref: px: {self._x_ref_lqr[0]*1000:.2f} py: {self._x_ref_lqr[4]*1000:.2f} ax: {np.rad2deg(self._x_ref_lqr[2]):.3f} ay: {np.rad2deg(self._x_ref_lqr[6]):.3f}")

    def reference_state(self) -> State:
        return State.from_iterable(self._x_ref_lqr)

    def _refresh_u_ref(self) -> None:
        self.u_ref_lqr = (self._u_ref_true @ self._x_ref_lqr).ravel()

    def _update_pos_integrator(self, state: State):
        
        angle_ok = np.hypot(state.ax, state.ay) < self.angle_thresh
        rate_ok  = np.hypot(state.wx, state.wy) < self.rate_thresh if self.rate_thresh else True
        pos_outside_thresh = (
            np.hypot(
                state.px - self._x_ref_true[0],
                state.py - self._x_ref_true[4],
            )
            > self.pos_thresh
        )

        if self._integrator_active:
            # looser condition to stay active: only angle and rate
            condition = angle_ok and rate_ok
        else:
            # tighter condition to activate: must also be off-center enough to bother
            condition = pos_outside_thresh and angle_ok and rate_ok
        
        if condition:
            self._integrator_active = True
            pos_err_x = state.px - self._x_ref_true[0]  # true center error
            pos_err_y = state.py - self._x_ref_true[4]
            # shift reference AWAY from current position
            # so LQR sees a bigger error and pushes harder toward true center
            self._x_ref_lqr[0] -= self.ki * pos_err_x * self.actuator_dt
            self._x_ref_lqr[4] -= self.ki * pos_err_y * self.actuator_dt

            # must follow x_ref_lqr
            self._refresh_u_ref()
            
            #self._print_ref()

    def _update_tilt_integrator(self, state: State):
        ax_stale = self._update_tilt_x_stale(state.ax)
        ay_stale = self._update_tilt_y_stale(state.ay)

        if self._tilt_x_integrator_active:
            ax_condition = abs(state.ax) > self.tilt_deadband
        else:
            ax_condition = ax_stale

        if self._tilt_y_integrator_active:
            ay_condition = abs(state.ay) > self.tilt_deadband
        else:
            ay_condition = ay_stale

        if ax_condition:
            self._tilt_x_integrator_active = True
            tilt_err_x = state.ax - self._x_ref_true[2]
            self._x_ref_lqr[2] -= self.tilt_ki * tilt_err_x * self.actuator_dt
            self._x_ref_lqr[2] = np.clip(
                self._x_ref_lqr[2],
                self._x_ref_true[2] - self.max_tilt_bias,
                self._x_ref_true[2] + self.max_tilt_bias,
            )
            self._refresh_u_ref()            

        if ay_condition:
            self._tilt_y_integrator_active = True
            tilt_err_y = state.ay - self._x_ref_true[6]
            self._x_ref_lqr[6] -= self.tilt_ki * tilt_err_y * self.actuator_dt
            self._x_ref_lqr[6] = np.clip(
                self._x_ref_lqr[6],
                self._x_ref_true[6] - self.max_tilt_bias,
                self._x_ref_true[6] + self.max_tilt_bias,
            )
            self._refresh_u_ref()
        
        if ay_condition or ax_condition:
            pass
            #self._print_ref()

    def _update_tilt_x_stale(self, angle: float) -> bool:
        if abs(angle) <= self.tilt_deadband:
            self._tilt_x_same_sign_time = 0.0
            self._tilt_x_prev_sign = 0.0
            return False

        sign = float(np.sign(angle))
        if sign == self._tilt_x_prev_sign:
            self._tilt_x_same_sign_time += self.actuator_dt
        else:
            self._tilt_x_same_sign_time = self.actuator_dt
            self._tilt_x_prev_sign = sign

        return self._tilt_x_same_sign_time >= self.tilt_stale_time_s

    def _update_tilt_y_stale(self, angle: float) -> bool:
        if abs(angle) <= self.tilt_deadband:
            self._tilt_y_same_sign_time = 0.0
            self._tilt_y_prev_sign = 0.0
            return False

        sign = float(np.sign(angle))
        if sign == self._tilt_y_prev_sign:
            self._tilt_y_same_sign_time += self.actuator_dt
        else:
            self._tilt_y_same_sign_time = self.actuator_dt
            self._tilt_y_prev_sign = sign

        return self._tilt_y_same_sign_time >= self.tilt_stale_time_s

    def compute(self, state: State) -> ControlInput:
        x_err = state.as_vector() - self._x_ref_lqr
        u_err = self._u_prev - self.u_ref_lqr
        xi_err = np.concatenate([x_err, u_err])

        delta_u = -(self.K @ xi_err).ravel()
        delta_u = self._limit_delta_u(delta_u)
        u = self._u_prev + delta_u
        u = self._limit_command_radius(u)

        return ControlInput(float(u[0]), float(u[1]))

    def set_applied_command(self, u: ControlInput, state: State) -> None:
        self._u_prev = np.array([u.px_cmd, u.py_cmd], dtype=float)
        self._update_pos_integrator(state)
        self._update_tilt_integrator(state)
        
    def reset(self, x_hat: State | None = None):
        print("delta_lqr reset", x_hat)

        # reset integrator
        self._x_ref_lqr = self._x_ref_true.copy()
        self.u_ref_lqr = (self._u_ref_true @ self._x_ref_lqr).ravel()
        self._integrator_active = False
        self._tilt_x_integrator_active = False
        self._tilt_y_integrator_active = False
        self._tilt_x_same_sign_time = 0.0
        self._tilt_y_same_sign_time = 0.0
        self._tilt_x_prev_sign = 0.0
        self._tilt_y_prev_sign = 0.0
        
        self._u_prev = self.u_ref_lqr.copy()
        if x_hat is not None:
            u = self.compute(x_hat)
            self._u_prev = np.array([u.px_cmd, u.py_cmd], dtype=float)


        
