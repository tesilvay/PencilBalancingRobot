from dataclasses import dataclass, field

import numpy as np

from src.shared import Measurement, WorkspaceParams, default_workspace

from .base import GainScheduler


@dataclass
class PowerGainScheduleParams:
    ang_gain: float = 1.0
    ang_power: float = 3.0
    workspace: WorkspaceParams = field(default_factory=default_workspace)


POWER_GAIN_SCHEDULE_PRESETS = {
    "default": {
        "ang_gain": 2.6,
        "ang_power": 1.5,
    },
    "light": {
        "ang_gain": 1.2,
        "ang_power": 1.08,
    },
    "s_to_l": {
        "ang_gain": 0.9,
        "ang_power": 0.94,
    }
}


class PowerGainScheduler(GainScheduler):
    def __init__(self, params: PowerGainScheduleParams):
        self.workspace = params.workspace
        self.ang_gain = float(params.ang_gain)
        self.ang_power = float(params.ang_power)

        if self.ang_power <= 0.0:
            raise ValueError("ang_power must be positive")

    @staticmethod
    def _shape(value: float, center: float, gain: float, power: float) -> float:
        delta = float(value - center)
        shaped = gain * np.sign(delta) * (np.abs(delta) ** power)
        return float(center + shaped)

    def apply(self, y_raw: Measurement) -> Measurement:
        return Measurement(
            px=float(y_raw.px),
            py=float(y_raw.py),
            ax=self._shape(y_raw.ax, 0.0, self.ang_gain, self.ang_power),
            ay=self._shape(y_raw.ay, 0.0, self.ang_gain, self.ang_power),
        )

    def map_angle(self, angle_rad: float) -> float:
        return self._shape(angle_rad, 0.0, self.ang_gain, self.ang_power)
