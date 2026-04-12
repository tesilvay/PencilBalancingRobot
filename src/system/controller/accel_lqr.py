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
)

from .base import BaseController


@dataclass
class AccelLQRParams:
    Q_single_axis: np.ndarray
    R: np.ndarray
    plant: PlantParams = field(default_factory=default_plant)
    timing: TimingParams = field(default_factory=default_timing)
    workspace: WorkspaceParams = field(default_factory=default_workspace)
    max_acc_cmd: float | None = None
    max_vel_cmd: float | None = None
    pos_correction_gain: float = 0.1
    vel_correction_gain: float = 0.001


ACCEL_LQR_PRESETS = {
    "default": {
        "Q_single_axis": np.diag([5.0, 0.1, 100.0, 1.0]),
        "R": np.eye(2) * 1e-1,
        "max_acc_cmd": 9.81 * 5,
        "max_vel_cmd": 0.35,
        "pos_correction_gain": 0.0,
        "vel_correction_gain": 0.0,
    },
    "drift_safe": {
        "base": "default",
        "pos_correction_gain": 1.0,
        "vel_correction_gain": 1.0,
    },
}


class AccelLQRController(BaseController):
    """
    Continuous-time LQR on the acceleration-input transformed plant.

    The controller computes table acceleration from the non-augmented
    controller-coordinate state, then integrates that acceleration into
    velocity and position commands while enforcing the same safety clamps used
    by the acceleration pole-placement controller.
    """

    def __init__(self, params: AccelLQRParams):
        from src.system.plant.dynamics_model import BuildAccModel

        model = BuildAccModel(params.plant)
        A = np.asarray(model["A_ctrl"], dtype=float)
        B = np.asarray(model["B_ctrl"], dtype=float)

        Q_block = np.asarray(params.Q_single_axis, dtype=float)
        Q = np.block([
            [Q_block, np.zeros_like(Q_block)],
            [np.zeros_like(Q_block), Q_block],
        ])
        R = np.asarray(params.R, dtype=float)

        self.K, _, _ = ct.lqr(A, B, Q, R)
        self.K = np.asarray(self.K, dtype=float)

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

        self._cmd_pos = np.array([self.x_ref, self.y_ref], dtype=float)
        self._cmd_vel = np.zeros(2, dtype=float)
        self._anti_windup_active = False

    def _controller_axis_state(
        self,
        pos_cmd: float,
        vel_cmd: float,
        angle: float,
        angle_vel: float,
    ) -> np.ndarray:
        return np.array([
            pos_cmd - self.l * angle,
            vel_cmd - self.l * angle_vel,
            -self.g * angle,
            -self.g * angle_vel,
        ], dtype=float)

    def _reference_error_vector(self, state: State) -> np.ndarray:
        x_ctrl = self._controller_axis_state(
            self._cmd_pos[0], self._cmd_vel[0], state.ax, state.wx
        )
        y_ctrl = self._controller_axis_state(
            self._cmd_pos[1], self._cmd_vel[1], state.ay, state.wy
        )

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

    def clamp_acc(self, acc_cmd: np.ndarray) -> np.ndarray:
        if self.max_acc_cmd is None:
            return acc_cmd

        acc_norm = float(np.linalg.norm(acc_cmd))
        if acc_norm > self.max_acc_cmd and acc_norm > 0.0:
            self._anti_windup_active = True
            acc_cmd = acc_cmd * (self.max_acc_cmd / acc_norm)
        return acc_cmd

    def clamp_vel(self, vel_cmd: np.ndarray) -> np.ndarray:
        if self.max_vel_cmd is None:
            return vel_cmd

        vel_norm = float(np.linalg.norm(vel_cmd))
        if vel_norm > self.max_vel_cmd and vel_norm > 0.0:
            self._anti_windup_active = True
            vel_cmd = vel_cmd * (self.max_vel_cmd / vel_norm)
        return vel_cmd

    def clamp_pos(self, pos_cmd: np.ndarray) -> np.ndarray:
        if self.safe_radius is None:
            return pos_cmd

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

    def _apply_drift_correction(self, state: State) -> None:
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

        return ControlInput(px_cmd=float(self._cmd_pos[0]), py_cmd=float(self._cmd_pos[1]))

    def set_applied_command(self, u: ControlInput, state: State) -> None:
        del state
        self._cmd_pos[0] = float(u.px_cmd)
        self._cmd_pos[1] = float(u.py_cmd)

    def reset(self, x_hat: State | None = None):
        if x_hat is None:
            self._cmd_pos[:] = [self.x_ref, self.y_ref]
            self._cmd_vel[:] = 0.0
            self._anti_windup_active = False
            return

        self._cmd_pos[:] = [float(x_hat.px), float(x_hat.py)]
        self._cmd_vel[:] = [float(x_hat.vx), float(x_hat.vy)]
        self._anti_windup_active = False
