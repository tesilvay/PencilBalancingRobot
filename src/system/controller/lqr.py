from dataclasses import dataclass

from core.sim_types import SystemState, TableCommand
import numpy as np
import control as ct

from src.shared import PlantParams
from .base import BaseController


@dataclass
class LQRParams:
    plant:          PlantParams
    Q_single_axis:  np.ndarray
    R:              np.ndarray 


LQR_PRESETS = {
    "default": {
        "plant": "default:default",
        "Q_single_axis": np.diag([0.01, 0.01, 100, 10]),
        "R":             np.eye(2) * 1e6,
    }
}


class LQRController(BaseController):

    def __init__(self, A, B, Q, R, x_ref=None):
        self.K, _, _ = ct.lqr(A, B, Q, R)
        self.x_ref = np.zeros(A.shape[0]) if x_ref is None else x_ref.as_vector()
        
        # compute steady-state feedforward
        self.u_ref = -np.linalg.pinv(B) @ (A @ self.x_ref)

    def compute(self, state):
        x = state.as_vector()
        
        error = x - self.x_ref

        u = self.u_ref - self.K @ (error)

        return TableCommand(u[0], u[1])
