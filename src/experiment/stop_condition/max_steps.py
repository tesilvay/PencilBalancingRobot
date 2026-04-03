from dataclasses import dataclass, field

from src.shared import TimingParams, default_timing
from .stabilized import StabilizedCondition, StabilizedParams


def _default_stabilized() -> StabilizedParams:
    return StabilizedParams(tol_ang_deg=10.0, tol_m=20e-3, settle_time=1.0, time_in_tol=0.0)


@dataclass
class MaxStepsConditionParams:
    timing:            TimingParams     = field(default_factory=default_timing)
    stabilized_params: StabilizedParams = field(default_factory=_default_stabilized)


MAX_STEPS_CONDITION_PRESETS = {
    "default": {}
}


class MaxStepsCondition(StabilizedCondition):
    def __init__(self, params: MaxStepsConditionParams):
        super().__init__(params.stabilized_params)
        self.steps = int(params.timing.total_time / params.timing.dt)

    def should_stop(self, i, state, dt):
        super().should_stop(i, state, dt)
        return i >= self.steps
