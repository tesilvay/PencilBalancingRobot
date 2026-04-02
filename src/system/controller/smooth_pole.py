from dataclasses import dataclass

from core.sim_types import SystemState, TableCommand
import numpy as np
import control as ct

from src.shared import PlantParams, TimingParams
from .base import BaseController


@dataclass
class SmoothPoleParams:
    plant:          PlantParams
    timing:         TimingParams
    s_poles:        list[float]
    slew_poles:     float 


SMOOTH_POLE_PRESETS = {
    "default": {
        "plant":         "default:default",
        "timing":        "default:default",
        "s_poles":       [-14, -16, -18, -20] * 2,
        "slew_poles":    0.95
    }
}


class SmoothPolePlacementController(BaseController):
    """Discrete-time Δu feedback via augmented state ξ = [x; u_{k-1}], v = Δu, gains from pole placement."""

    def __init__(
        self,
        A_c: np.ndarray,
        B_c: np.ndarray,
        dt: float,
        desired_poles_z: np.ndarray,
        x_ref: SystemState | None = None,
    ):
        n, m = A_c.shape[0], B_c.shape[1]
        sys_c = ct.ss(A_c, B_c, np.eye(n), np.zeros((n, m)))
        sys_d = ct.c2d(sys_c, dt)
        A_d = np.array(sys_d.A)
        B_d = np.array(sys_d.B)

        z = np.asarray(desired_poles_z, dtype=complex).ravel()
        if z.size != n + m:
            raise ValueError(
                f"desired_poles_z must have length {n + m} (dim ξ), got {z.size}"
            )

        A_aug = np.block([[A_d, B_d], [np.zeros((m, n)), np.eye(m)]])
        B_aug = np.vstack([B_d, np.eye(m)])
        self.K = ct.place(A_aug, B_aug, z)

        self.x_ref = np.zeros(n) if x_ref is None else x_ref.as_vector()
        self.u_ref = (-np.linalg.pinv(B_c) @ (A_c @ self.x_ref)).ravel()
        self.xi_ref = np.concatenate([self.x_ref, self.u_ref])
        self._u_prev = self.u_ref.copy()

    def compute(self, state: SystemState) -> TableCommand:
        x = state.as_vector()
        xi = np.concatenate([x, self._u_prev])
        v = -(self.K @ (xi - self.xi_ref)).ravel()
        u = self._u_prev + v
        return TableCommand(float(u[0]), float(u[1]))

    def set_applied_command(self, cmd: TableCommand) -> None:
        self._u_prev = np.array([cmd.x_des, cmd.y_des], dtype=float)

    def reset(self):
        self._u_prev = self.u_ref.copy()
