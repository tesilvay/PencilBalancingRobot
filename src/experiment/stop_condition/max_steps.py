from dataclasses import dataclass

from src.shared import TimingParams
from .stabilized import StabilizedCondition, StabilizedParams


@dataclass
class MaxStepsConditionParams:
    timing:      TimingParams
    stabilized_params: StabilizedParams



MAX_STEPS_CONDITION_PRESETS = {
    "default": {
        "timing":      "default:default",
        "stabilized_params": "default:lazy",
    }
}


class MaxStepsCondition(StabilizedCondition):
    def __init__(self, params: MaxStepsConditionParams):
        p = params
        super().__init__(p.stabilized_params)
        self.steps = int(p.timing.total_time / p.timing.dt)

    def should_stop(self, i, state, dt):
        super().should_stop(i, state, dt)
        return i >= self.steps
