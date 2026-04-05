from dataclasses import dataclass, field

import numpy as np

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
class CircleParams:
    period_s:  float
    radius:    float           = 0.03
    plant:     PlantParams     = field(default_factory=default_plant)
    timing:    TimingParams    = field(default_factory=default_timing)
    workspace: WorkspaceParams = field(default_factory=default_workspace)


CIRCLE_PRESETS = {
    "default": {
        "period_s": 18,
        "radius":   0.03,
    }
}


class CircleController(BaseController):

    def __init__(self, params: CircleParams):
        x_ref = make_reference_state(params.workspace)

        self.x_ref    = x_ref
        self.radius   = params.radius
        self.period_s = params.period_s
        self.omega    = 2 * np.pi / params.period_s
        self.dt       = params.timing.dt
        self.t        = 0.0

    def compute(self, state: State) -> ControlInput:
        self.t += self.dt

        cx = self.x_ref.px
        cy = self.x_ref.py

        x = cx + self.radius * np.cos(self.omega * self.t)
        y = cy + self.radius * np.sin(self.omega * self.t)

        return ControlInput(x, y)

    def reset(self):
        self.t = 0.0
