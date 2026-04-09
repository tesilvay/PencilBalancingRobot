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
    s_poles:    list[float]
    slew_poles: float
    plant:      PlantParams     = field(default_factory=default_plant)
    timing:     TimingParams    = field(default_factory=default_timing)
    workspace:  WorkspaceParams = field(default_factory=default_workspace)


SMOOTH_POLE_PRESETS = {
    "default": {
        "s_poles":    [-12, -14, -16, -18] * 2,
        "slew_poles": 0.99,
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
        A_c, B_c = BuildLinearModel(params.plant)
        dt        = params.timing.dt
        x_ref     = make_reference_state(params.workspace)

        n, m = A_c.shape[0], B_c.shape[1]
        sys_c = ct.ss(A_c, B_c, np.eye(n), np.zeros((n, m)))
        sys_d = ct.c2d(sys_c, dt)
        A_d   = np.array(sys_d.A)
        B_d   = np.array(sys_d.B)

        # Augmented-system poles: n s-domain → z + m slew poles in z
        z_from_s = np.exp(np.array(params.s_poles, dtype=complex) * dt)
        z_slew   = np.full(m, params.slew_poles, dtype=complex)
        z        = np.concatenate([z_from_s, z_slew])

        if z.size != n + m:
            raise ValueError(
                f"desired poles must have length {n + m} (dim ξ), got {z.size}"
            )

        A_aug = np.block([[A_d, B_d], [np.zeros((m, n)), np.eye(m)]])
        B_aug = np.vstack([B_d, np.eye(m)])
        self.K = ct.place(A_aug, B_aug, z)

        self.x_ref  = x_ref.as_vector()
        self.u_ref  = (-np.linalg.pinv(B_c) @ (A_c @ self.x_ref)).ravel()
        self.xi_ref = np.concatenate([self.x_ref, self.u_ref])
        self._u_prev = self.u_ref.copy()

    def compute(self, state: State) -> ControlInput:
        x   = state.as_vector()
        xi  = np.concatenate([x, self._u_prev])
        v   = -(self.K @ (xi - self.xi_ref)).ravel()
        u   = self._u_prev + v
        return ControlInput(float(u[0]), float(u[1]))

    def set_applied_command(self, u: ControlInput) -> None:
        self._u_prev = np.array([u.px_cmd, u.py_cmd], dtype=float)

    def reset(self, x_hat: State | None = None):
        if x_hat is None:
            self._u_prev = self.u_ref.copy()
            return

        # Warm-start internal command memory from the estimated state at switch.
        self._u_prev = self.u_ref.copy()
        u = self.compute(x_hat)
        self._u_prev = np.array([u.px_cmd, u.py_cmd], dtype=float)
