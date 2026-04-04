from dataclasses import dataclass, field

import numpy as np
import control as ct

from src.shared import (
    PlantParams,
    WorkspaceParams,
    State,
    ControlInput,
    default_plant,
    default_workspace,
    make_reference_state,
)

from .base import BaseController


@dataclass
class LQRParams:
    Q_single_axis: np.ndarray
    R:             np.ndarray
    plant:         PlantParams     = field(default_factory=default_plant)
    workspace:     WorkspaceParams = field(default_factory=default_workspace)


LQR_PRESETS = {
    "default": {
        "Q_single_axis": np.diag([0.01, 0.01, 100, 10]),
        "R":             np.eye(2) * 1e6,
    }
}


class LQRController(BaseController):

    def __init__(self, params: LQRParams):
        from src.system.plant.dynamics_model import BuildLinearModel
        A, B = BuildLinearModel(params.plant)
        x_ref = make_reference_state(params.workspace)

        Q_block = params.Q_single_axis
        Q = np.block([
            [Q_block,               np.zeros_like(Q_block)],
            [np.zeros_like(Q_block), Q_block              ],
        ])

        self.K, _, _ = ct.lqr(A, B, Q, params.R)
        self.x_ref = x_ref.as_vector()
        self.u_ref = -np.linalg.pinv(B) @ (A @ self.x_ref)

    def compute(self, state: State) -> ControlInput:
        x     = state.as_vector()
        error = x - self.x_ref
        u     = self.u_ref - self.K @ error
        return ControlInput(u[0], u[1])
