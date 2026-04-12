"""Shared discrete-time linear dynamics and measurement map for estimators."""

import numpy as np
import control as ct

from src.shared import PlantParams, TimingParams
from src.system.plant.dynamics_model import BuildLinearModel


def _continuous_AB(
    plant: PlantParams,
    mode: str,
    *,
    placing_anchor_stiffness: float = 400.0,
    placing_anchor_damping: float = 40.0,
    placing_gravity_scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "free":
        return BuildLinearModel(plant)
    if mode != "placing":
        raise ValueError(f"Unknown estimator dynamics mode: {mode!r}")

    tau = float(plant.tau)
    zeta = float(plant.zeta)
    l = float(plant.com_length)
    k_anchor = float(placing_anchor_stiffness)
    c_anchor = float(placing_anchor_damping)
    gravity_scale = float(placing_gravity_scale)
    alpha_coeff = gravity_scale * float(plant.g) / l - k_anchor

    A_axis = np.array([
        [0.0,             1.0,              0.0,         0.0],
        [-1.0 / tau**2,   -2.0 * zeta/tau,  0.0,         0.0],
        [0.0,             0.0,              0.0,         1.0],
        [1.0/(l*tau**2),  2.0*zeta/(l*tau), alpha_coeff, -c_anchor],
    ])
    B_axis = np.array([
        [0.0],
        [1.0 / tau**2],
        [0.0],
        [-1.0 / (l * tau**2)],
    ])

    z4 = np.zeros((4, 4))
    z4x1 = np.zeros((4, 1))
    A = np.block([
        [A_axis, z4],
        [z4, A_axis],
    ])
    B = np.block([
        [B_axis, z4x1],
        [z4x1, B_axis],
    ])
    return A, B


def discretize_AB(
    plant: PlantParams,
    timing: TimingParams | float,
    mode: str = "free",
    *,
    placing_anchor_stiffness: float = 400.0,
    placing_anchor_damping: float = 40.0,
    placing_gravity_scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    dt = float(timing.dt if isinstance(timing, TimingParams) else timing)
    A_c, B_c = _continuous_AB(
        plant,
        mode=mode,
        placing_anchor_stiffness=placing_anchor_stiffness,
        placing_anchor_damping=placing_anchor_damping,
        placing_gravity_scale=placing_gravity_scale,
    )
    sys_c = ct.ss(A_c, B_c, np.eye(8), np.zeros((8, 2)))
    sys_d = ct.c2d(sys_c, dt)
    return np.array(sys_d.A), np.array(sys_d.B)


def measurement_H() -> np.ndarray:
    # z = [px, ax, py, ay] = H x,  x = [px, vx, ax, wx, py, vy, ay, wy]
    H = np.zeros((4, 8))
    H[0, 0] = 1.0
    H[1, 2] = 1.0
    H[2, 4] = 1.0
    H[3, 6] = 1.0
    return H
