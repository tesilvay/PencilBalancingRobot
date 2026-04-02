from dataclasses import dataclass
from numpy.linalg import norm

from .base import Supervisor


@dataclass
class DynamicSupervisorParams:
    stable_threshold:  float
    stable_hold_s:     float
    consistent_hold_s: float
    loss_threshold:    float


DYNAMIC_SUPERVISOR_PRESETS = {
    "default": {
        "stable_threshold":  0.035,
        "stable_hold_s":     2.0,
        "consistent_hold_s": 1.0,
        "loss_threshold":    0.3,
    }
}


class DynamicSupervisor(Supervisor):
    def __init__(self, params: DynamicSupervisorParams):
        self.params  = params
        self.state   = "ACQUISITION"
        self._t_state  = 0.0
        self._t_stable = 0.0
        self._t_lost   = 0.0

    def update(self, x_est, innovation, dt) -> tuple[str, str]:
        self._t_state += dt
        self._step(x_est, innovation, dt)
        return self._active()

    def _step(self, x_est, innovation, dt):
        if self._is_stable(x_est):  self._t_stable += dt
        else:                        self._t_stable  = 0.0

        if self._is_lost(innovation): self._t_lost += dt
        else:                          self._t_lost  = 0.0

        s = self.state
        if s == "ACQUISITION":
            if self._t_stable >= self.params.stable_hold_s:
                self._transition("STABILIZATION_READY")

        elif s == "STABILIZATION_READY":
            if self._t_lost >= self.params.loss_hold_s:
                self._transition("ACQUISITION")
            elif self._t_state >= self.params.consistent_hold_s:
                self._transition("STABILIZING")

        elif s == "STABILIZING":
            if self._t_lost >= self.params.loss_hold_s:
                self._transition("ACQUISITION")
            elif self._t_state >= self.params.stable_hold_s:
                self._transition("BALANCED")

        elif s == "BALANCED":
            if self._t_lost >= self.params.loss_hold_s:
                self._transition("ACQUISITION")

    def _transition(self, new_state):
        self.state    = new_state
        self._t_state = 0.0

    def _active(self) -> tuple[str, str]:
        return {
            "ACQUISITION":         ("follower", "lpf"),
            "STABILIZATION_READY": ("follower", "lpf"),
            "STABILIZING":         ("smooth",   "kalman"),
            "BALANCED":            ("full",     "kalman"),
        }[self.state]

    def _is_stable(self, x_est):   return norm(x_est[:2]) < self.params.stable_threshold
    def _is_lost(self, innovation): return innovation is not None and norm(innovation) > self.params.loss_threshold
