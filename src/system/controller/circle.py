from dataclasses import dataclass
import numpy as np

from src.shared import PlantParams, TimingParams, SystemState, TableCommand


@dataclass
class CircleParams:
    plant:          PlantParams
    timing:         TimingParams
    period_s:       float


CIRCLE_PRESETS = {
    "default": {
        "plant":         "default:default",
        "timing":        "default:default",
        "period_s":       18,
    }
}


class CircleController:
    def __init__(self, x_ref: SystemState, radius: float, period_s: float, dt:float):
        self.x_ref = x_ref
        self.radius = radius
        self.period_s = period_s
        self.omega = 2 * np.pi / period_s
        self.t = 0.0
        self.dt = dt

    def compute(self, state):
        self.t += self.dt

        cx = self.x_ref.x
        cy = self.x_ref.y

        x = cx + self.radius * np.cos(self.omega * self.t)
        y = cy + self.radius * np.sin(self.omega * self.t)

        return TableCommand(x, y)

    def reset(self):
        self.t = 0.0
