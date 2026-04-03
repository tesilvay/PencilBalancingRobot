from dataclasses import dataclass, field

import numpy as np
import control as ct

from src.shared import (
    PlantParams,
    WorkspaceParams,
    SystemState,
    TableCommand,
    default_plant,
    default_workspace,
    make_reference_state,
)

from .base import BaseController


@dataclass
class PoleParams:
    poles:     list[float]
    plant:     PlantParams     = field(default_factory=default_plant)
    workspace: WorkspaceParams = field(default_factory=default_workspace)


POLE_PRESETS = {
    "default": {
        "poles": [-14, -16, -18, -20] * 2,
    }
}


class PolePlacementController(BaseController):

    def __init__(self, params: PoleParams):
        from src.system.plant.dynamics_model import BuildLinearModel
        A, B = BuildLinearModel(params.plant)
        x_ref = make_reference_state(params.workspace)

        self.K    = ct.place(A, B, params.poles)
        self.x_ref = x_ref.as_vector()
        self.u_ref = -np.linalg.pinv(B) @ (A @ self.x_ref)

    def compute(self, state: SystemState) -> TableCommand:
        x     = state.as_vector()
        error = x - self.x_ref
        u     = self.u_ref - self.K @ error
        return TableCommand(u[0], u[1])
