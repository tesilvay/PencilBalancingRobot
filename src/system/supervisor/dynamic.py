from dataclasses import dataclass
import numpy as np
from numpy.linalg import norm

from .base import Supervisor


@dataclass
class DynamicSupervisorParams:
    stable_threshold:  float
    stable_hold_s:     float
    consistent_hold_s: float
    loss_threshold:    float
    loss_hold_s:       float
    match_pos_threshold: float
    match_vel_threshold: float
    match_ang_threshold: float
    match_w_threshold: float
    match_hold_s: float
    blend_ramp_s: float


DYNAMIC_SUPERVISOR_PRESETS = {
    "default": {
        "stable_threshold":  0.035,
        "stable_hold_s":     2.0,
        "consistent_hold_s": 1.0,
        "loss_threshold":    0.3,
        "loss_hold_s":       0.5,
        "match_pos_threshold": 8e-3,
        "match_vel_threshold": 60e-3,
        "match_ang_threshold": np.deg2rad(2.0),
        "match_w_threshold": np.deg2rad(25.0),
        "match_hold_s": 0.35,
        "blend_ramp_s": 2.0,
    }
}


class DynamicSupervisor(Supervisor):
    def __init__(self, params: DynamicSupervisorParams):
        self.params  = params
        self.state   = "ACQUISITION"
        self._t_state  = 0.0
        self._t_stable = 0.0
        self._t_lost   = 0.0
        self._t_match = 0.0
        self._t_blend = 0.0
        self._est_k = 0.0
        self._blend_active = False
        self._last_transition: dict | None = None
        self._offset_latched = False

    def update(
        self,
        x_hat_0,
        innovation_0,
        x_hat_1,
        innovation_1,
        dt,
    ) -> tuple[int, float]:
        prev_state = self.state
        self._t_state += dt
        self._step(x_hat_0, innovation_0, x_hat_1, innovation_1, dt)
        self._last_transition = {
            "prev_state": prev_state,
            "new_state": self.state,
            "left_acquisition": (prev_state == "ACQUISITION" and self.state != "ACQUISITION"),
        }
        if self._last_transition["left_acquisition"]:
            self._offset_latched = True
        return self.active_output

    @property
    def active_output(self) -> tuple[int, float]:
        return self._controller_index(), self._est_k

    @property
    def is_offset_latched(self) -> bool:
        return self._offset_latched

    @property
    def state_name(self) -> str:
        return self.state

    @property
    def last_transition(self) -> dict | None:
        return self._last_transition

    def _step(self, x_hat_0, innovation_0, x_hat_1, innovation_1, dt):
        x_used = self._blend_state(x_hat_0, x_hat_1)
        innovation_used = self._blend_innovation(innovation_0, innovation_1)

        if self._is_stable(x_used):  self._t_stable += dt
        else:                        self._t_stable  = 0.0

        if self._is_lost(innovation_used): self._t_lost += dt
        else:                               self._t_lost  = 0.0

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
            else:
                self._update_blend(x_hat_0, x_hat_1, dt)
                if self._est_k >= 1.0:
                    self._transition("BALANCED")

        elif s == "BALANCED":
            if self._t_lost >= self.params.loss_hold_s:
                self._transition("ACQUISITION")
            else:
                self._est_k = 1.0

    def _transition(self, new_state):
        self.state    = new_state
        self._t_state = 0.0
        if new_state == "STABILIZING":
            self._reset_blend_progress()
        elif new_state == "BALANCED":
            self._est_k = 1.0
            self._blend_active = False
        elif new_state == "ACQUISITION":
            self._reset_blend_progress()

    def _controller_index(self) -> int:
        return {
            "ACQUISITION": 0,
            "STABILIZATION_READY": 1,
            "STABILIZING": 1,
            "BALANCED": 1,
        }[self.state]

    def _reset_blend_progress(self) -> None:
        self._t_match = 0.0
        self._t_blend = 0.0
        self._est_k = 0.0
        self._blend_active = False

    def _update_blend(self, x_hat_0, x_hat_1, dt: float) -> None:
        if not self._estimators_match(x_hat_0, x_hat_1):
            self._reset_blend_progress()
            return

        if not self._blend_active:
            self._t_match += dt
            if self._t_match < self.params.match_hold_s:
                self._est_k = 0.0
                return
            self._blend_active = True
            self._t_blend = 0.0

        self._t_blend += dt
        ramp_s = max(float(self.params.blend_ramp_s), 1e-9)
        self._est_k = min(self._t_blend / ramp_s, 1.0)

    def _estimators_match(self, x_hat_0, x_hat_1) -> bool:
        dx = np.abs(x_hat_0.as_vector() - x_hat_1.as_vector())
        thresholds = np.array([
            self.params.match_pos_threshold,
            self.params.match_vel_threshold,
            self.params.match_ang_threshold,
            self.params.match_w_threshold,
            self.params.match_pos_threshold,
            self.params.match_vel_threshold,
            self.params.match_ang_threshold,
            self.params.match_w_threshold,
        ], dtype=float)
        return bool(np.all(dx <= thresholds))

    def _blend_state(self, x_hat_0, x_hat_1):
        blended = (1.0 - self._est_k) * x_hat_0.as_vector() + self._est_k * x_hat_1.as_vector()
        return type(x_hat_0).from_iterable(blended)

    def _blend_innovation(self, innovation_0, innovation_1):
        if innovation_0 is None and innovation_1 is None:
            return None
        if innovation_0 is None:
            return innovation_1
        if innovation_1 is None:
            return innovation_0
        a = np.asarray(innovation_0, dtype=float)
        b = np.asarray(innovation_1, dtype=float)
        return (1.0 - self._est_k) * a + self._est_k * b

    def _is_stable(self, x_est):   return norm([x_est.px, x_est.py]) < self.params.stable_threshold
    def _is_lost(self, innovation): return innovation is not None and norm(innovation) > self.params.loss_threshold

    def reset(self):
        self.state = "ACQUISITION"
        self._t_state = 0.0
        self._t_stable = 0.0
        self._t_lost = 0.0
        self._reset_blend_progress()
        self._last_transition = None
        self._offset_latched = False
