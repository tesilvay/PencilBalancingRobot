from dataclasses import dataclass, field

import numpy as np

from src.shared import Measurement, WorkspaceParams, default_workspace

from .base import GainScheduler


@dataclass
class DeadzoneGainScheduleParams:
    ang_deadzone: float = np.deg2rad(1.0)
    workspace: WorkspaceParams = field(default_factory=default_workspace)


DEADZONE_GAIN_SCHEDULE_PRESETS = {
    "default": {
        "ang_deadzone": np.deg2rad(1.0),
    }
}


class DeadzoneGainScheduler(GainScheduler):
    def __init__(self, params: DeadzoneGainScheduleParams):
        self.workspace = params.workspace
        self.ang_deadzone = float(params.ang_deadzone)

        if self.ang_deadzone < 0.0:
            raise ValueError("ang_deadzone must be non-negative")

    @staticmethod
    def _shape(value: float, center: float, deadzone: float) -> float:
        delta = float(value - center)
        if abs(delta) < deadzone:
            return float(center)
        return float(center + delta - np.sign(delta) * deadzone)

    def apply(self, y_raw: Measurement) -> Measurement:
        return Measurement(
            px=float(y_raw.px),
            py=float(y_raw.py),
            ax=self._shape(y_raw.ax, 0.0, self.ang_deadzone),
            ay=self._shape(y_raw.ay, 0.0, self.ang_deadzone),
        )

    def map_angle(self, angle_rad: float) -> float:
        return self._shape(angle_rad, 0.0, self.ang_deadzone)
