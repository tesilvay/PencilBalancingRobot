from dataclasses import dataclass, field

import numpy as np

from src.shared import (
    ControlInput,
    State,
    WorkspaceParams,
    default_workspace,
)

from .base import BaseController


@dataclass
class SimpleControllerParams:
    kc: float
    kcv: float
    k_alpha: float
    k_omega: float
    max_tilt_ref_deg: float | None = None
    max_delta_u: float | None = None
    workspace: WorkspaceParams = field(default_factory=default_workspace)


SIMPLE_CONTROLLER_PRESETS = {
    "default": {
        "kc": 4.0,
        "kcv": 0.0,
        "k_alpha": 9.0e-0,
        "k_omega": 0.0e-4,
        
        "max_tilt_ref_deg": 3.0,
        "max_delta_u": None,
    },
    "gentle": {
        "base": "default",
        "k_alpha": 2.5e-3,
        "k_omega": 2.5e-4,
        "max_delta_u": 2.5e-4,
    },
    "stronger": {
        "base": "default",
        "k_alpha": 1.0e-2,
        "k_omega": 1.0e-3,
        "max_delta_u": 1.0e-3,
    },
    "quadratic_ref": {
        "kc": 4.5, # or 0.3 in normal
        "kcv": 0.0,
        "k_alpha": 1.0e-2,
        "k_omega": 0.0e-4,
        
        "max_tilt_ref_deg": 2.0,
        "max_delta_u": None,
    },
}


class SimpleController(BaseController):
    """Direct non-model-based tilt controller for independent x/y axes."""

    def __init__(self, params: SimpleControllerParams):
        self._params = params
        self._max_tilt_ref = (
            None
            if params.max_tilt_ref_deg is None
            else float(np.deg2rad(params.max_tilt_ref_deg))
        )
        self._u_prev = np.array(
            [float(params.workspace.x_ref), float(params.workspace.y_ref)],
            dtype=float,
        )
        #self._plot_tilt_reference_vector_field()

    def _plot_tilt_reference(self) -> None:
        import matplotlib.pyplot as plt

        pos_cm = np.linspace(0.0, 7.0, 200)
        pos_m = pos_cm / 100.0
        tilt_ref_deg = np.rad2deg([self._tilt_ref(pos, 0.0) for pos in pos_m])

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(pos_cm, tilt_ref_deg, linewidth=2.0)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_title("Simple Controller Tilt Reference")
        ax.set_xlabel("Position from center (cm)")
        ax.set_ylabel("Tilt reference (deg)")
        ax.set_xlim(0.0, 7.0)
        ax.set_ylim(-3.0, 0.0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show(block=False)

    def _plot_tilt_reference_vector_field(self) -> None:
        import matplotlib.pyplot as plt

        pos_cm = np.linspace(-7.0, 7.0, 15)
        x_cm, y_cm = np.meshgrid(pos_cm, pos_cm)
        x_m = x_cm / 100.0
        y_m = y_cm / 100.0

        alpha_x_deg = np.rad2deg(
            np.vectorize(lambda pos: self._tilt_ref(float(pos), 0.0))(x_m)
        )
        alpha_y_deg = np.rad2deg(
            np.vectorize(lambda pos: self._tilt_ref(float(pos), 0.0))(y_m)
        )
        magnitude = np.sqrt(alpha_x_deg * alpha_x_deg + alpha_y_deg * alpha_y_deg)

        fig, ax = plt.subplots(figsize=(6, 6))
        quiver = ax.quiver(
            x_cm,
            y_cm,
            alpha_x_deg,
            alpha_y_deg,
            magnitude,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.004,
        )
        fig.colorbar(quiver, ax=ax, label="Tilt reference magnitude (deg)")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_title("Simple Controller Tilt Reference Vector Field")
        ax.set_xlabel("X position from center (cm)")
        ax.set_ylabel("Y position from center (cm)")
        ax.set_xlim(-7.5, 7.5)
        ax.set_ylim(-7.5, 7.5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show(block=False)

    def _tilt_ref(self, pos: float, vel: float) -> float:
        alpha_ref = -self._params.kc * pos * abs(pos) - self._params.kcv * vel
        if self._max_tilt_ref is None:
            return float(alpha_ref)
        return float(np.clip(alpha_ref, -self._max_tilt_ref, self._max_tilt_ref))

    def _axis_delta_command(self, pos: float, vel: float, alpha: float, omega: float) -> float:
        alpha_ref = self._tilt_ref(pos, vel)
        return float(
            self._params.k_alpha * (alpha - alpha_ref) * abs(alpha - alpha_ref)
            + self._params.k_omega * omega
        )

    def _limit_delta_u(self, delta_u: np.ndarray) -> np.ndarray:
        max_delta_u = self._params.max_delta_u
        if max_delta_u is None:
            return delta_u

        max_delta = float(max_delta_u)
        if max_delta <= 0.0:
            return np.zeros_like(delta_u)

        delta_norm = float(np.linalg.norm(delta_u))
        if delta_norm <= max_delta or delta_norm <= 0.0:
            return delta_u

        return delta_u * (max_delta / delta_norm)

    def compute(self, state: State) -> ControlInput:
        delta_u = np.array(
            [
                self._axis_delta_command(state.px, state.vx, state.ax, state.wx),
                self._axis_delta_command(state.py, state.vy, state.ay, state.wy),
            ],
            dtype=float,
        )
        delta_u = self._limit_delta_u(delta_u)
        u = self._u_prev + delta_u
        return ControlInput(float(u[0]), float(u[1]))

    def set_applied_command(self, u: ControlInput) -> None:
        self._u_prev = np.array([u.px_cmd, u.py_cmd], dtype=float)

    def reset(self, x_hat: State | None = None):
        del x_hat
        self._u_prev = np.array(
            [
                float(self._params.workspace.x_ref),
                float(self._params.workspace.y_ref),
            ],
            dtype=float,
        )
