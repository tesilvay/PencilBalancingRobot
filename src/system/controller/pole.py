from dataclasses import dataclass

from src.shared import PlantParams, SystemState, TableCommand
import numpy as np
import control as ct

from .base import BaseController


@dataclass
class PoleParams:
    plant:  PlantParams
    poles:  list[float]


POLE_PRESETS = {
    "default": {
        "plant": "default:default",
        "poles": [-14, -16, -18, -20] * 2,
    }
}


class PolePlacementController(BaseController):

    def __init__(self, A, B, desired_poles, x_ref=None):
        self.K = ct.place(A, B, desired_poles)

        self.x_ref = np.zeros(A.shape[0]) if x_ref is None else x_ref.as_vector()

        # compute steady-state feedforward
        self.u_ref = -np.linalg.pinv(B) @ (A @ self.x_ref)

    def compute(self, state):
        x = state.as_vector()
        
        error = x - self.x_ref

        u = self.u_ref - self.K @ (error)

        return TableCommand(u[0], u[1])
