from dataclasses import dataclass, field

from src.shared import Measurement, WorkspaceParams, default_workspace

from .base import GainScheduler


@dataclass
class LinearCubicGainScheduleParams:
    ang_gain: float = 1.0
    ang_cubic_gain: float = 0.0
    workspace: WorkspaceParams = field(default_factory=default_workspace)


LINEAR_CUBIC_GAIN_SCHEDULE_PRESETS = {
    "default": {
        "ang_gain": 1.0,
        "ang_cubic_gain": 0.0,
    }
}


class LinearCubicGainScheduler(GainScheduler):
    def __init__(self, params: LinearCubicGainScheduleParams):
        self.workspace = params.workspace
        self.ang_gain = float(params.ang_gain)
        self.ang_cubic_gain = float(params.ang_cubic_gain)

    @staticmethod
    def _shape(value: float, center: float, linear_gain: float, cubic_gain: float) -> float:
        delta = float(value - center)
        return float(center + linear_gain * delta + cubic_gain * delta ** 3)

    def apply(self, y_raw: Measurement) -> Measurement:
        return Measurement(
            px=float(y_raw.px),
            py=float(y_raw.py),
            ax=self._shape(y_raw.ax, 0.0, self.ang_gain, self.ang_cubic_gain),
            ay=self._shape(y_raw.ay, 0.0, self.ang_gain, self.ang_cubic_gain),
        )

    def map_angle(self, angle_rad: float) -> float:
        return self._shape(angle_rad, 0.0, self.ang_gain, self.ang_cubic_gain)
