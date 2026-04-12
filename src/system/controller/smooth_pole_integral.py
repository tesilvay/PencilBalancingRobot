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
class SmoothPoleIntegralParams:
    max_pole: float
    pole_step: float
    slew_poles: float
    int_pole: float
    int_pole_step: float
    int_angle_threshold: float
    max_eta: float
    plant: PlantParams = field(default_factory=default_plant)
    timing: TimingParams = field(default_factory=default_timing)
    workspace: WorkspaceParams = field(default_factory=default_workspace)


SMOOTH_POLE_INTEGRAL_PRESETS = {
    "default": {
        "max_pole": -6,
        "pole_step": 1,
        "slew_poles": 0.99,
        "int_pole": 0.999,
        "int_pole_step": 0.002,
        "int_angle_threshold": 1.5,
        "max_eta": 0.02,
    },
    "lead": {
        "max_pole": -9,
        "pole_step": 1,
        "slew_poles": 0.986,
        "int_pole": 0.995,
        "int_pole_step": 0.002,
        "int_angle_threshold": 3,
        "max_eta": 0.02,
    },
    "stronger": {
        "base": "default",
        "int_pole": 0.993,
        "int_pole_step": 0.002,
    },
    "gentle": {
        "base": "default",
        "int_pole": 0.998,
        "int_pole_step": 0.001,
    },
}


class SmoothPoleIntegralController(BaseController):
    """Smooth Δu pole placement with added integral states on px/py error."""

    def __init__(self, params: SmoothPoleIntegralParams):
        from src.system.plant.dynamics_model import BuildLinearModel

        A_c, B_c = BuildLinearModel(params.plant)
        dt = float(params.timing.dt)
        self.dt = dt
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
        z_int = np.array(
            [
                params.int_pole,
                params.int_pole - params.int_pole_step,
            ],
            dtype=complex,
        )
        z = np.concatenate([z_from_s, z_slew, z_int])

        A_aug = np.block([[A_d, B_d], [np.zeros((m, n)), np.eye(m)]])
        B_aug = np.vstack([B_d, np.eye(m)])

        idx_px = 0
        idx_py = 4
        C_pos = np.zeros((2, n))
        C_pos[0, idx_px] = 1.0
        C_pos[1, idx_py] = 1.0
        C_int = dt * C_pos

        n_aug = n + m
        n_int = 2
        A_full = np.block([
            [A_aug, np.zeros((n_aug, n_int))],
            [C_int, np.zeros((n_int, m)), np.eye(n_int)],
        ])
        B_full = np.block([
            [B_aug],
            [np.zeros((n_int, m))],
        ])

        if z.size != n_aug + n_int:
            raise ValueError(
                f"desired poles must have length {n_aug + n_int} (dim chi), got {z.size}"
            )

        ctrb_rank = np.linalg.matrix_rank(ct.ctrb(A_full, B_full))
        if ctrb_rank != n_aug + n_int:
            raise ValueError(
                f"integral augmented system is not controllable: rank {ctrb_rank}, expected {n_aug + n_int}"
            )

        self.K = ct.place(A_full, B_full, z)

        self.x_ref = x_ref.as_vector()
        self.u_ref = (-np.linalg.pinv(B_c) @ (A_c @ self.x_ref)).ravel()
        self.xi_ref = np.concatenate([self.x_ref, self.u_ref])
        self.chi_ref = np.concatenate([self.xi_ref, np.zeros(n_int)])
        self.C_pos = C_pos
        self.int_angle_threshold = float(np.deg2rad(params.int_angle_threshold))
        self.max_eta = float(params.max_eta)
        self._u_prev = self.u_ref.copy()
        self._eta = np.zeros(n_int, dtype=float)

        p_full_ol = np.linalg.eigvals(A_full)
        p_full_cl = np.linalg.eigvals(A_full - B_full @ self.K)

        print("Integral augmented open-loop poles:")
        print(p_full_ol)
        print("Magnitudes:", np.abs(p_full_ol))

        print("\nIntegral augmented closed-loop poles:")
        print(p_full_cl)
        print("Magnitudes:", np.abs(p_full_cl))
        print("\nRequested poles:")
        print(z)

    def compute(self, state: State) -> ControlInput:
        x = state.as_vector()
        pos_err = self.C_pos @ (x - self.x_ref)
        angle_err = abs(float(state.ax)) + abs(float(state.ay))
        if angle_err < self.int_angle_threshold:
            self._eta = np.clip(
                self._eta + self.dt * pos_err,
                -self.max_eta,
                self.max_eta,
            )

        chi = np.concatenate([x, self._u_prev, self._eta])
        v = -(self.K @ (chi - self.chi_ref)).ravel()
        u = self._u_prev + v
        return ControlInput(float(u[0]), float(u[1]))

    def set_applied_command(self, u: ControlInput, state: State) -> None:
        del state
        self._u_prev = np.array([u.px_cmd, u.py_cmd], dtype=float)

    def reset(self, x_hat: State | None = None):
        self._eta = np.zeros(2, dtype=float)

        if x_hat is None:
            self._u_prev = self.u_ref.copy()
            return

        self._u_prev = self.u_ref.copy()
        u = self.compute(x_hat)
        self._u_prev = np.array([u.px_cmd, u.py_cmd], dtype=float)
        self._eta = np.zeros(2, dtype=float)
