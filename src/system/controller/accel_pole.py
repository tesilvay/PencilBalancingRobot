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
    poles: list[float]
    plant: PlantParams = field(default_factory=default_plant)
    timing: TimingParams = field(default_factory=default_timing)
    workspace: WorkspaceParams = field(default_factory=default_workspace)
    max_acc_cmd: float | None = None
    pos_correction_gain: float = 0.1
    vel_correction_gain: float = 0.1


ACCEL_POLE_PRESETS = {
    "default": {
        "poles": [-1.0, -3.0, -4.0, -5.0, -6.0] * 2,
        "pos_correction_gain": 0.001,
        "vel_correction_gain": 0.001,
    },
    "drift_safe": {
        "base": "default",
        "pos_correction_gain": 1.0,
        "vel_correction_gain": 1.0,
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
        A_aug = model["A_aug"]
        B_aug = model["B_aug"]

        if len(params.poles) != A_aug.shape[0]:
            raise ValueError(
                f"expected {A_aug.shape[0]} poles for acceleration controller, "
                f"got {len(params.poles)}"
            )

        self.K = np.asarray(ct.place(A_aug, B_aug, params.poles), dtype=float)

        self.g = params.plant.g
        self.l = params.plant.com_length
        self.dt = float(params.timing.dt)
        self.max_acc_cmd = (
            float(params.max_acc_cmd)
            if params.max_acc_cmd is not None
            else params.plant.max_acc
        )

        self.x_ref = float(params.workspace.x_ref)
        self.y_ref = float(params.workspace.y_ref)
        self.pos_correction_gain = float(params.pos_correction_gain)
        self.vel_correction_gain = float(params.vel_correction_gain)

        self._int_error = np.zeros(2, dtype=float)
        self._cmd_pos = np.array([self.x_ref, self.y_ref], dtype=float)
        self._cmd_vel = np.zeros(2, dtype=float)

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

        self._int_error[0] += (self.x_ref - x_ctrl[0]) * self.dt
        self._int_error[1] += (self.y_ref - y_ctrl[0]) * self.dt

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

    def _apply_drift_correction(self, state: State):
        self._cmd_pos[0] += self.pos_correction_gain * (state.px - self._cmd_pos[0]) * self.dt
        self._cmd_pos[1] += self.pos_correction_gain * (state.py - self._cmd_pos[1]) * self.dt
        self._cmd_vel[0] += self.vel_correction_gain * (state.vx - self._cmd_vel[0]) * self.dt
        self._cmd_vel[1] += self.vel_correction_gain * (state.vy - self._cmd_vel[1]) * self.dt

    def compute(self, state: State) -> ControlInput:
        error_vec = self._reference_error_vector(state)
        acc_cmd = -np.asarray(self.K @ error_vec, dtype=float).reshape(-1)

        if self.max_acc_cmd is not None:
            norm = float(np.linalg.norm(acc_cmd))
            if norm > self.max_acc_cmd and norm > 0.0:
                acc_cmd *= self.max_acc_cmd / norm

        self._cmd_pos += self._cmd_vel * self.dt + 0.5 * acc_cmd * self.dt * self.dt
        self._cmd_vel += acc_cmd * self.dt
        self._apply_drift_correction(state)

        return ControlInput(px_cmd=float(self._cmd_pos[0]), py_cmd=float(self._cmd_pos[1]))

    def set_applied_command(self, u: ControlInput) -> None:
        self._cmd_pos[0] = float(u.px_cmd)
        self._cmd_pos[1] = float(u.py_cmd)

    def reset(self, x_hat: State | None = None):
        self._int_error[:] = 0.0

        if x_hat is None:
            self._cmd_pos[:] = [self.x_ref, self.y_ref]
            self._cmd_vel[:] = 0.0
            return

        self._cmd_pos[:] = [float(x_hat.px), float(x_hat.py)]
        self._cmd_vel[:] = [float(x_hat.vx), float(x_hat.vy)]
