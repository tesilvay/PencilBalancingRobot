from dataclasses import dataclass, field

import numpy as np
import control as ct

from src.shared import (
    ControlInput,
    PlantParams,
    State,
    TimingParams,
    WorkspaceParams,
    default_plant,
    default_timing,
    default_workspace,
)

from .base import BaseController


@dataclass
class AccelPoleParams:
    max_pole: float
    pole_step: float
    plant: PlantParams = field(default_factory=default_plant)
    timing: TimingParams = field(default_factory=default_timing)
    workspace: WorkspaceParams = field(default_factory=default_workspace)
    max_acc_cmd: float | None = None
    max_vel_cmd: float | None = None
    pos_correction_gain: float = 0.1
    vel_correction_gain: float = 0.1
    discrete_time: bool = True
    use_integrator: bool = False


ACCEL_POLE_PRESETS = {
    "default": {
        "max_pole": -4.0,
        "pole_step": 1,
        
        "max_acc_cmd": 9.81*5,
        "max_vel_cmd": 0.50,
        "pos_correction_gain": 0.1,
        "vel_correction_gain": 0.001,
        "discrete_time": True,
        "use_integrator": False,
    },
    "test0": {
        "base": "default",
        "poles": [0.99, 0.995, 0.998, 0.999] * 2,
    },
    "test0int": {
        "base": "default",
        "poles": [0.99, 0.995, 0.998, 0.999, 0.9989] * 2,
    },
    "test01": {
        "base": "default",
        "poles": [0.9, 0.92, 0.94, 0.96] * 2,
    },
    "test1": {
        "base": "default",
        "poles": [0.94, 0.95, 0.96, 0.97] * 2,
    },
    "test2": {
        "base": "default",
        "poles": [-2.0, -3.0, -4.0, -5.0] * 2,
    },
    "test3": {
        "base": "default",
        "poles": [-3.0, -4.0, -5.0, -6.0] * 2,
    },
    "drift_safe": {
        "base": "default",
        "pos_correction_gain": 1.0,
        "vel_correction_gain": 1.0,
    },
    "continuous_integral_test0": {
        "base": "default",
        "poles": [-0.1, -0.2, -0.4, -0.8, -1.2] * 2,
        "discrete_time": False,
        "use_integrator": True,
    },
}


class AccelPolePlacementController(BaseController):
    """
    Pole-placement controller on the acceleration-input transformed plant.

    Internally it computes table acceleration, integrates it into a commanded
    position trajectory, and returns that trajectory through the existing
    ControlInput interface.
    """

    def __init__(self, params: AccelPoleParams):
        from src.system.plant.dynamics_model import BuildAccModel

        model = BuildAccModel(params.plant)
        self.use_integrator = bool(params.use_integrator)
        self.discrete_time = bool(params.discrete_time)

        if self.use_integrator:
            A = np.asarray(model["A_aug"], dtype=float)
            B = np.asarray(model["B_aug"], dtype=float)
        else:
            A = np.asarray(model["A_ctrl"], dtype=float)
            B = np.asarray(model["B_ctrl"], dtype=float)

        n, m = A.shape[0], B.shape[1]
        if self.discrete_time:
            sys = ct.c2d(ct.ss(A, B, np.eye(n), np.zeros((n, m))), params.timing.dt)
            A_used = np.asarray(sys.A, dtype=float)
            B_used = np.asarray(sys.B, dtype=float)
        else:
            A_used = A
            B_used = B

        if params.max_pole is None:
            self.K = np.zeros((m, n), dtype=float)
        else:
            poles_list = [
                params.max_pole - i * params.pole_step
                for i in range(n // 2)
            ] * 2
            poles = np.asarray(poles_list, dtype=complex)
            if poles.size != n:
                raise ValueError(
                    f"expected {n} poles for acceleration controller, got {poles.size}"
                )

            if self.discrete_time:
                if np.all(np.abs(poles) < 1.0):
                    placed_poles = poles
                else:
                    placed_poles = np.exp(poles * params.timing.dt)
            else:
                placed_poles = poles

            self.K = np.asarray(ct.place(A_used, B_used, placed_poles), dtype=float)

        self.g = params.plant.g
        self.l = params.plant.com_length
        self.dt = float(params.timing.dt)
        self.max_acc_cmd = params.max_acc_cmd
        self.max_vel_cmd = params.max_vel_cmd

        self.x_ref = float(params.workspace.x_ref)
        self.y_ref = float(params.workspace.y_ref)
        self.safe_radius = params.workspace.safe_radius
        
        self.pos_correction_gain = float(params.pos_correction_gain)
        self.vel_correction_gain = float(params.vel_correction_gain)

        self._int_error = np.zeros(2, dtype=float)
        self._cmd_pos = np.array([self.x_ref, self.y_ref], dtype=float)
        self._cmd_vel = np.zeros(2, dtype=float)
        self._anti_windup_active = False

    def _controller_axis_state(self, pos_cmd: float, vel_cmd: float, angle: float, angle_vel: float):
        return np.array([
            pos_cmd - self.l * angle,
            vel_cmd - self.l * angle_vel,
            -self.g * angle,
            -self.g * angle_vel,
        ], dtype=float)

    def _reference_error_vector(self, state: State):
        x_ctrl = self._controller_axis_state(self._cmd_pos[0], self._cmd_vel[0], state.ax, state.wx)
        y_ctrl = self._controller_axis_state(self._cmd_pos[1], self._cmd_vel[1], state.ay, state.wy)

        if self.use_integrator:
            return np.array([
                self._int_error[0],
                x_ctrl[0] - self.x_ref,
                x_ctrl[1],
                x_ctrl[2],
                x_ctrl[3],
                self._int_error[1],
                y_ctrl[0] - self.y_ref,
                y_ctrl[1],
                y_ctrl[2],
                y_ctrl[3],
            ], dtype=float)

        return np.array([
            x_ctrl[0] - self.x_ref,
            x_ctrl[1],
            x_ctrl[2],
            x_ctrl[3],
            y_ctrl[0] - self.y_ref,
            y_ctrl[1],
            y_ctrl[2],
            y_ctrl[3],
        ], dtype=float)

    def _integrate_error(self, state: State) -> None:
        x_ctrl = self._controller_axis_state(self._cmd_pos[0], self._cmd_vel[0], state.ax, state.wx)
        y_ctrl = self._controller_axis_state(self._cmd_pos[1], self._cmd_vel[1], state.ay, state.wy)
        self._int_error[0] += (self.x_ref - x_ctrl[0]) * self.dt
        self._int_error[1] += (self.y_ref - y_ctrl[0]) * self.dt

    def clamp_acc(self, acc_cmd: np.ndarray) -> np.ndarray:
        acc_norm = float(np.linalg.norm(acc_cmd))
        if acc_norm > self.max_acc_cmd and acc_norm > 0.0:
            self._anti_windup_active = True
            acc_cmd = acc_cmd * (self.max_acc_cmd / acc_norm)
        return acc_cmd

    def clamp_vel(self, vel_cmd: np.ndarray) -> np.ndarray:
        vel_norm = float(np.linalg.norm(vel_cmd))
        if vel_norm > self.max_vel_cmd and vel_norm > 0.0:
            self._anti_windup_active = True
            vel_cmd = vel_cmd * (self.max_vel_cmd / vel_norm)
        return vel_cmd

    def clamp_pos(self, pos_cmd: np.ndarray) -> np.ndarray:
        dx = float(pos_cmd[0] - self.x_ref)
        dy = float(pos_cmd[1] - self.y_ref)
        dist = float(np.sqrt(dx * dx + dy * dy))

        if dist > self.safe_radius and dist > 0.0:
            self._anti_windup_active = True
            scale = self.safe_radius / dist
            pos_cmd = np.array([
                self.x_ref + dx * scale,
                self.y_ref + dy * scale,
            ], dtype=float)
        return pos_cmd

    def _apply_drift_correction(self, state: State):
        self._cmd_pos[0] += self.pos_correction_gain * (state.px - self._cmd_pos[0]) * self.dt
        self._cmd_pos[1] += self.pos_correction_gain * (state.py - self._cmd_pos[1]) * self.dt
        self._cmd_vel[0] += self.vel_correction_gain * (state.vx - self._cmd_vel[0]) * self.dt
        self._cmd_vel[1] += self.vel_correction_gain * (state.vy - self._cmd_vel[1]) * self.dt

    def compute(self, state: State) -> ControlInput:
        self._anti_windup_active = False
        error_vec = self._reference_error_vector(state)
        
        acc_cmd = -np.asarray(self.K @ error_vec, dtype=float).reshape(-1)
        acc_cmd = self.clamp_acc(acc_cmd)

        self._cmd_vel += acc_cmd * self.dt
        self._cmd_vel = self.clamp_vel(self._cmd_vel)
        
        self._cmd_pos += self._cmd_vel * self.dt
        self._cmd_pos = self.clamp_pos(self._cmd_pos)
        
        self._apply_drift_correction(state)
        self._cmd_vel = self.clamp_vel(self._cmd_vel)
        self._cmd_pos = self.clamp_pos(self._cmd_pos)

        if self.use_integrator and not self._anti_windup_active:
            self._integrate_error(state)

        return ControlInput(px_cmd=float(self._cmd_pos[0]), py_cmd=float(self._cmd_pos[1]))

    def set_applied_command(self, u: ControlInput) -> None:
        self._cmd_pos[0] = float(u.px_cmd)
        self._cmd_pos[1] = float(u.py_cmd)

    def reset(self, x_hat: State | None = None):
        self._int_error[:] = 0.0

        if x_hat is None:
            self._cmd_pos[:] = [self.x_ref, self.y_ref]
            self._cmd_vel[:] = 0.0
            self._anti_windup_active = False
            return

        self._cmd_pos[:] = [float(x_hat.px), float(x_hat.py)]
        self._cmd_vel[:] = [float(x_hat.vx), float(x_hat.vy)]
        self._anti_windup_active = False
