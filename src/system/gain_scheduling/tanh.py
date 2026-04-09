from dataclasses import dataclass, field

import numpy as np

from src.shared import Measurement, WorkspaceParams, default_workspace

from .base import GainScheduler


@dataclass
class TanhGainScheduleParams:
    ang_gain: float = 1.0
    ang_scale: float = np.deg2rad(5.0)
    workspace: WorkspaceParams = field(default_factory=default_workspace)


TANH_GAIN_SCHEDULE_PRESETS = {
    "default": {
        "ang_gain": 1.0,
        "ang_scale": np.deg2rad(5.0),
    }
}


class TanhGainScheduler(GainScheduler):
    def __init__(self, params: TanhGainScheduleParams):
        self.workspace = params.workspace
        self.ang_gain = float(params.ang_gain)
        self.ang_scale = float(params.ang_scale)

        if self.ang_scale <= 0.0:
            raise ValueError("ang_scale must be positive")

    @staticmethod
    def _shape(value: float, center: float, gain: float, scale: float) -> float:
        delta = float(value - center)
        return float(center + scale * np.tanh(gain * delta / scale))

    def apply(self, y_raw: Measurement) -> Measurement:
        return Measurement(
            px=float(y_raw.px),
            py=float(y_raw.py),
            ax=self._shape(y_raw.ax, 0.0, self.ang_gain, self.ang_scale),
            ay=self._shape(y_raw.ay, 0.0, self.ang_gain, self.ang_scale),
        )

    def map_angle(self, angle_rad: float) -> float:
        return self._shape(angle_rad, 0.0, self.ang_gain, self.ang_scale)
