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
class AccelLagPoleParams:
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


ACCEL_LAG_POLE_PRESETS = {
    "default": {
        "max_pole": 0.9999,
        "pole_step": 0.005,
        "max_acc_cmd": 9.81 * 5,
        "max_vel_cmd": 0.25,
        "pos_correction_gain": 0.001,
        "vel_correction_gain": 0.001,
        "discrete_time": True,
    },
    "test0": {
        "base": "default",
        "poles": [0.985, 0.99, 0.992, 0.995] * 2,
    },
    "test1": {
        "base": "default",
        "poles": [0.95, 0.96, 0.97, 0.98] * 2,
    },
    "test2": {
        "base": "default",
        "poles": [0.92, 0.94, 0.96, 0.98] * 2,
    },
    "test3": {
        "base": "default",
        "poles": [0.82, 0.84, 0.86, 0.88] * 2,
    },
}


class AccelLagPolePlacementController(BaseController):
    """
    Lag-aware acceleration-style pole-placement controller.

    The feedback is designed on the lag-aware plant, but the controller still
    produces a virtual acceleration-like correction that is integrated into the
    commanded table trajectory returned through ControlInput.
    """

    def __init__(self, params: AccelLagPoleParams):
        from src.system.plant.dynamics_model import BuildAccModelWithLag

        model = BuildAccModelWithLag(params.plant)
        A = np.asarray(model["A_ctrl"], dtype=float)
        B = np.asarray(model["B_ctrl"], dtype=float)

        n, m = A.shape[0], B.shape[1]
        if params.discrete_time:
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
                    f"expected {n} poles for lag-aware acceleration controller, got {poles.size}"
                )

            if params.discrete_time:
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
        self.x_ref = float(params.workspace.x_ref)
        self.y_ref = float(params.workspace.y_ref)
        self.safe_radius = params.workspace.safe_radius
        self.x_ctrl_ref = np.array([
            self.x_ref, 0.0, 0.0, 0.0,
            self.y_ref, 0.0, 0.0, 0.0,
        ], dtype=float)
        self.u_ref = (-np.linalg.pinv(B) @ (A @ self.x_ctrl_ref)).ravel()
        self._u_prev = self.u_ref.copy()

    def _controller_axis_state(self, pos: float, vel: float, angle: float, angle_vel: float):
        return np.array([
            pos - self.l * angle,
            vel - self.l * angle_vel,
            -self.g * angle,
            -self.g * angle_vel,
        ], dtype=float)

    def _reference_error_vector(self, state: State) -> np.ndarray:
        x_ctrl = self._controller_axis_state(state.px, state.vx, state.ax, state.wx)
        y_ctrl = self._controller_axis_state(state.py, state.vy, state.ay, state.wy)
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

    def clamp_pos(self, pos_cmd: np.ndarray) -> np.ndarray:
        if self.safe_radius is None:
            return pos_cmd

        dx = float(pos_cmd[0] - self.x_ref)
        dy = float(pos_cmd[1] - self.y_ref)
        dist = float(np.sqrt(dx * dx + dy * dy))

        if dist > self.safe_radius and dist > 0.0:
            scale = self.safe_radius / dist
            pos_cmd = np.array([
                self.x_ref + dx * scale,
                self.y_ref + dy * scale,
            ], dtype=float)
        return pos_cmd

    def compute(self, state: State) -> ControlInput:
        error_vec = self._reference_error_vector(state)
        u_cmd = self.u_ref - np.asarray(self.K @ error_vec, dtype=float).reshape(-1)
        u_cmd = self.clamp_pos(u_cmd)
        return ControlInput(px_cmd=float(u_cmd[0]), py_cmd=float(u_cmd[1]))

    def set_applied_command(self, u: ControlInput) -> None:
        self._u_prev[:] = [float(u.px_cmd), float(u.py_cmd)]

    def reset(self, x_hat: State | None = None):
        if x_hat is None:
            self._u_prev = self.u_ref.copy()
            return

        self._u_prev[:] = [float(x_hat.px), float(x_hat.py)]
