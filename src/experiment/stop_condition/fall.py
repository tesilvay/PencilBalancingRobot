from dataclasses import dataclass
from numpy import deg2rad

from .base import StopCondition


@dataclass
class FallConditionParams:
    max_angle_deg: float


FALL_CONDITION_PRESETS = {
    "default": {
        "max_angle_deg": 45.0,
    }
}


class FallCondition(StopCondition):
    def __init__(self, params: FallConditionParams):
        p = params
        self.max_angle = deg2rad(p.max_angle_deg)

    def should_stop(self, i, state, dt):
        return (
            abs(state.ax) > self.max_angle
            or abs(state.ay) > self.max_angle
        )
