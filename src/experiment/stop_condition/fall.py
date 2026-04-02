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
    def __init__(self, max_angle=deg2rad(45)):
        self.max_angle = max_angle

    def should_stop(self, i, state, dt):
        return (
            abs(state.alpha_x) > self.max_angle
            or abs(state.alpha_y) > self.max_angle
        )
