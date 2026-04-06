"""Shared discrete-time linear dynamics and measurement map for estimators."""

import numpy as np
import control as ct

from src.shared import PlantParams, TimingParams
from src.system.plant.dynamics_model import BuildLinearModel


def discretize_AB(plant: PlantParams, timing: TimingParams) -> tuple[np.ndarray, np.ndarray]:
    A_c, B_c = BuildLinearModel(plant)
    sys_c = ct.ss(A_c, B_c, np.eye(8), np.zeros((8, 2)))
    sys_d = ct.c2d(sys_c, timing.dt)
    return np.array(sys_d.A), np.array(sys_d.B)


def measurement_H() -> np.ndarray:
    # z = [px, ax, py, ay] = H x,  x = [px, vx, ax, wx, py, vy, ay, wy]
    H = np.zeros((4, 8))
    H[0, 0] = 1.0
    H[1, 2] = 1.0
    H[2, 4] = 1.0
    H[3, 6] = 1.0
    return H
