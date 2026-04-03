from dataclasses import dataclass
from numpy import deg2rad

from .base import StopCondition


@dataclass
class StabilizedParams:
    tol_ang_deg: float
    tol_m:       float
    settle_time: float
    time_in_tol: float


STABILIZED_CONDITION_PRESETS = {
    "default": {
        "tol_ang_deg": 5.0,
        "tol_m":       10e-3,
        "settle_time": 1.0,
        "time_in_tol": 0.0,
    },
    "lazy": {
        "base": "default",
        "tol_ang_deg": 10.0,
        "tol_m":       20e-3,
    }
}


class StabilizedCondition(StopCondition):
    def __init__(self, params: StabilizedParams):
        p = params
        
        self.tol_ang = deg2rad(p.tol_ang_deg)
        self.tol_m = p.tol_m
        self.settle_time = p.settle_time
        self.time_in_tol = p.time_in_tol
        
        self._stabilized = False
        self._settling_time = None

    def reset(self):
        self.time_in_tol = 0.0
        self._stabilized = False
        self._settling_time = None
    
    def _is_inside_tolerance(self, state):
        return (
            abs(state.ax) < self.tol_ang
            and abs(state.ay) < self.tol_ang
            and abs(state.px) < self.tol_m
            and abs(state.py) < self.tol_m
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
