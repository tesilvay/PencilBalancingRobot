from dataclasses import dataclass

import numpy as np
from numpy.linalg import norm

from .base import Supervisor


@dataclass
class DynamicSupervisorParams:
    stable_threshold: float
    stable_hold_s: float
    consistent_hold_s: float
    loss_threshold: float
    loss_hold_s: float
    match_pos_threshold: float
    match_vel_threshold: float
    match_ang_threshold: float
    match_w_threshold: float
    match_hold_s: float
    blend_ramp_s: float


DYNAMIC_SUPERVISOR_PRESETS = {
    "default": {
        "stable_threshold": 0.035,
        "stable_hold_s": 0.0,
        "consistent_hold_s": 2.0,
        "loss_threshold": 0.3,
        "loss_hold_s": 0.5,
        "match_pos_threshold": 8e-3,
        "match_vel_threshold": 60e-3,
        "match_ang_threshold": np.deg2rad(2.0),
        "match_w_threshold": np.deg2rad(25.0),
        "match_hold_s": 0.0,
        "blend_ramp_s": 0.0,
    }
}


class DynamicSupervisor(Supervisor):
    def __init__(self, params: DynamicSupervisorParams):
        self.params = params
        self.state = "stabilization_ready"
        self._t_state = 0.0
        self._t_stable = 0.0
        self._t_lost = 0.0
        self._est_k = 0.0
        self._last_transition: dict | None = None
        self._offset_latched = False

    @property
    def active_output(self) -> tuple[int, float]:
        return 0, self._est_k

    @property
    def is_offset_latched(self) -> bool:
        return self._offset_latched

    @property
    def is_ready_to_run(self) -> bool:
        return self.state == "stabilizing"

    @property
    def state_name(self) -> str:
        return self.state

    @property
    def last_transition(self) -> dict | None:
        return self._last_transition

    def update(
        self,
        x_hat_0,
        innovation_0,
        x_hat_1,
        innovation_1,
        dt,
    ) -> tuple[int, float]:
        del x_hat_1, innovation_1
        prev_state = self.state
        prev_prestart = self.is_prestart_state
        self._t_state += dt
        self._step(x_hat_0, innovation_0, dt)
        self._last_transition = {
            "prev_state": prev_state,
            "new_state": self.state,
            "left_prestart": prev_prestart and not self.is_prestart_state,
            "left_acquisition": prev_prestart and not self.is_prestart_state,
        }
        if self._last_transition["left_prestart"]:
            self._offset_latched = True
        return self.active_output

    def _step(self, x_hat_0, innovation_0, dt: float) -> None:
        if self._is_stable(x_hat_0):
            self._t_stable += dt
        else:
            self._t_stable = 0.0

        if self._is_lost(innovation_0):
            self._t_lost += dt
        else:
            self._t_lost = 0.0

        if self.state == "stabilization_ready":
            ready_hold = max(float(self.params.consistent_hold_s), 0.0)
            stable_hold = max(float(self.params.stable_hold_s), 0.0)
            if self._t_lost >= self.params.loss_hold_s:
                self._transition("stabilization_ready")
                return
            if self._t_state >= ready_hold and self._t_stable >= stable_hold:
                self._transition("stabilizing")
            return

        if self._t_lost >= self.params.loss_hold_s:
            self._transition("stabilization_ready")
            return

        self._est_k = 1.0

    def _transition(self, new_state: str) -> None:
        if new_state == self.state:
            self._t_state = 0.0
            if new_state == "stabilization_ready":
                self._est_k = 0.0
                self._offset_latched = False
            return

        self.state = new_state
        self._t_state = 0.0
        if new_state == "stabilization_ready":
            self._est_k = 0.0
            self._offset_latched = False
        elif new_state == "stabilizing":
            self._est_k = 1.0

    def _is_stable(self, x_est) -> bool:
        return norm([x_est.px, x_est.py]) < self.params.stable_threshold

    def _is_lost(self, innovation) -> bool:
        return innovation is not None and norm(innovation) > self.params.loss_threshold

    def reset(self):
        self.state = "stabilization_ready"
        self._t_state = 0.0
        self._t_stable = 0.0
        self._t_lost = 0.0
        self._est_k = 0.0
        self._last_transition = None
        self._offset_latched = False
