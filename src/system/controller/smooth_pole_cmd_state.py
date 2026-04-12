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
class SmoothPoleCommandStateParams:
    s_max_pole: float
    s_pole_step: float
    slew_poles: float
    plant: PlantParams = field(default_factory=default_plant)
    timing: TimingParams = field(default_factory=default_timing)
    workspace: WorkspaceParams = field(default_factory=default_workspace)
    pos_drift_gain: float = 0.0


SMOOTH_POLE_CMD_STATE_PRESETS = {
    "default": {
        "s_max_pole": -8,
        "s_pole_step": 1,
        "slew_poles": 0.988,
        "pos_drift_gain": 0.5,
    },
    "drift": {
        "base": "default",
        "pos_drift_gain": 0.05,
    },
}


class SmoothPoleCommandStateController(BaseController):
    """Smooth pole-placement using command-derived table motion coordinates."""

    def __init__(self, params: SmoothPoleCommandStateParams):
        from src.system.plant.dynamics_model import BuildAccModelWithLag

        model = BuildAccModelWithLag(params.plant)
        A_c = np.asarray(model["A_ctrl"], dtype=float)
        B_c = np.asarray(model["B_ctrl"], dtype=float)
        self.dt = float(params.timing.dt)
        self.g = float(params.plant.g)
        self.l = float(params.plant.com_length)
        self._cmd_vel = np.zeros(2, dtype=float)
        x_ref_phys = make_reference_state(params.workspace)
        self.x_ref = self._controller_state(
            x_ref_phys,
            np.array([x_ref_phys.px, x_ref_phys.py], dtype=float),
            np.zeros(2, dtype=float),
        )

        n, m = A_c.shape[0], B_c.shape[1]
        sys_c = ct.ss(A_c, B_c, np.eye(n), np.zeros((n, m)))
        sys_d = ct.c2d(sys_c, self.dt)
        A_d = np.array(sys_d.A)
        B_d = np.array(sys_d.B)
        
        s_poles = [
            params.s_max_pole - i * params.s_pole_step
            for i in range(n // 2)
        ] * 2

        z_from_s = np.exp(np.array(s_poles, dtype=complex) * self.dt)
        z_slew = np.full(m, params.slew_poles, dtype=complex)
        z = np.concatenate([z_from_s, z_slew])

        if z.size != n + m:
            raise ValueError(
                f"desired poles must have length {n + m} (dim xi), got {z.size}"
            )

        A_aug = np.block([[A_d, B_d], [np.zeros((m, n)), np.eye(m)]])
        B_aug = np.vstack([B_d, np.eye(m)])
        self.K = ct.place(A_aug, B_aug, z)

        self.u_ref = (-np.linalg.pinv(B_c) @ (A_c @ self.x_ref)).ravel()
        self.xi_ref = np.concatenate([self.x_ref, self.u_ref])

        self.pos_drift_gain = float(params.pos_drift_gain)
        self._u_prev = self.u_ref.copy()
        self._u_prev_prev = self.u_ref.copy()
        
        p_aug_ol = np.linalg.eigvals(A_aug)
        p_aug_cl = np.linalg.eigvals(A_aug - B_aug @ self.K)

        print("Augmented open-loop poles:")
        print(p_aug_ol)
        print("Magnitudes:", np.abs(p_aug_ol))

        print("\nAugmented closed-loop poles:")
        print(p_aug_cl)
        print("Magnitudes:", np.abs(p_aug_cl))

    def _controller_axis_state(
        self,
        pos_cmd: float,
        vel_cmd: float,
        angle: float,
        angle_vel: float,
    ) -> np.ndarray:
        return np.array(
            [
                pos_cmd - self.l * angle,
                vel_cmd - self.l * angle_vel,
                -self.g * angle,
                -self.g * angle_vel,
            ],
            dtype=float,
        )

    def _controller_state(
        self,
        state: State,
        cmd_pos: np.ndarray | None = None,
        cmd_vel: np.ndarray | None = None,
    ) -> np.ndarray:
        if cmd_pos is None:
            cmd_pos = self._u_prev
        if cmd_vel is None:
            cmd_vel = self._cmd_vel

        x_ctrl = self._controller_axis_state(
            float(cmd_pos[0]),
            float(cmd_vel[0]),
            float(state.ax),
            float(state.wx),
        )
        y_ctrl = self._controller_axis_state(
            float(cmd_pos[1]),
            float(cmd_vel[1]),
            float(state.ay),
            float(state.wy),
        )
        return np.concatenate([x_ctrl, y_ctrl])

    def compute(self, state: State) -> ControlInput:
        x_ctrl = self._controller_state(state)
        xi = np.concatenate([x_ctrl, self._u_prev])
        v = -(self.K @ (xi - self.xi_ref)).ravel()
        u = self._u_prev + v

        if self.pos_drift_gain != 0.0:
            meas_pos = np.array([state.px, state.py], dtype=float)
            u = u + self.pos_drift_gain * (meas_pos - self._u_prev) * self.dt

        return ControlInput(float(u[0]), float(u[1]))

    def set_applied_command(self, u: ControlInput, state: State) -> None:
        del state
        next_u = np.array([u.px_cmd, u.py_cmd], dtype=float)
        self._cmd_vel = (next_u - self._u_prev) / self.dt
        self._u_prev_prev = self._u_prev.copy()
        self._u_prev = next_u

    def reset(self, x_hat: State | None = None):
        if x_hat is None:
            self._u_prev = self.u_ref.copy()
            self._u_prev_prev = self.u_ref.copy()
            self._cmd_vel = np.zeros(2, dtype=float)
            return

        pos = np.array([float(x_hat.px), float(x_hat.py)], dtype=float)
        self._u_prev = pos.copy()
        self._u_prev_prev = pos.copy()
        self._cmd_vel = np.array([float(x_hat.vx), float(x_hat.vy)], dtype=float)
