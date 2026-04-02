from dataclasses import dataclass

from src.shared import TimingParams
from .stabilized import StabilizedCondition


@dataclass
class MaxStepsConditionParams:
    timing:      TimingParams
    tol_ang_deg: float
    tol_m:       float
    settle_time: float


MAX_STEPS_CONDITION_PRESETS = {
    "default": {
        "timing":      "default:default",
        "tol_ang_deg": 10.0,
        "tol_m":       10e-3,
        "settle_time": 0.5,
    }
}


class MaxStepsCondition(StabilizedCondition):
    def __init__(self, steps, tol_ang, tol_m, settle_time):
        super().__init__(tol_ang=tol_ang, tol_m=tol_m, settle_time=settle_time)
        self.steps = steps

    def should_stop(self, i, state, dt):
        super().should_stop(i, state, dt)
        return i >= self.steps
