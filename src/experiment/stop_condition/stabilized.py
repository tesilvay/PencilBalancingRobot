from dataclasses import dataclass

from .base import StopCondition


@dataclass
class StabilizedParams:
    tol_ang_deg: float
    tol_m:       float
    settle_time: float


STABILIZED_CONDITION_PRESETS = {
    "default": {
        "tol_ang_deg": 10.0,
        "tol_m":       10e-3,
        "settle_time": 0.5,
    }
}


class StabilizedCondition(StopCondition):
    def __init__(self, tol_ang, tol_m, settle_time):
        self.tol_ang = tol_ang
        self.tol_m = tol_m
        self.settle_time = settle_time
        self.time_in_tol = 0.0
        self._stabilized = False
        self._settling_time = None

    def reset(self):
        self.time_in_tol = 0.0
        self._stabilized = False
        self._settling_time = None
    
    def _is_inside_tolerance(self, state):
        return (
            abs(state.alpha_x) < self.tol_ang
            and abs(state.alpha_y) < self.tol_ang
            and abs(state.x) < self.tol_m
            and abs(state.y) < self.tol_m
        )

    def should_stop(self, i, state, dt):
        if (self._is_inside_tolerance(state)):
            self.time_in_tol += dt
        else:
            self.time_in_tol = 0.0

        if (not self._stabilized) and self.time_in_tol >= self.settle_time:
            self._stabilized = True
            self._settling_time = i * dt
            return True

        return False

    def is_stabilized(self):
        return self._stabilized
    
    def settling_time(self):
        return self._settling_time
