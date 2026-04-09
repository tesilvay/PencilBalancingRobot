from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from src.shared import WorkspaceParams, default_workspace

from .base import RealServoSupervisorBase, RealStartupParams


@dataclass(kw_only=True)
class RealDynamicSupervisorParams(RealStartupParams):
    match_pos_threshold: float
    match_vel_threshold: float
    match_ang_threshold: float
    match_w_threshold: float
    match_hold_s: float
    blend_ramp_s: float
    workspace: WorkspaceParams = field(default_factory=default_workspace)


REAL_DYNAMIC_SUPERVISOR_PRESETS = {
    "default": {
        "centering_controller_index": 0,
        "run_controller_index": 1,
        
        "stable_threshold_deg": 4.0,
        "stable_threshold_m": 20e-3,
        "stable_hold_s": 1.0,
        
        "match_pos_threshold": 6e-3,
        "match_vel_threshold": 200e-3,
        "match_ang_threshold": np.deg2rad(1.5),
        "match_w_threshold": np.deg2rad(35.0),
        "match_hold_s": 0.35,
        "blend_ramp_s": 2.0,
        
        "manual_step_m": 0.002,
    }
}


class RealDynamicSupervisor(RealServoSupervisorBase):
    def __init__(self, params: RealDynamicSupervisorParams):
        super().__init__(params)
        self._t_state = 0.0
        self._t_match = 0.0
        self._t_blend = 0.0
        self._est_k = 0.0
        self._blend_active = False

    @property
    def active_output(self) -> tuple[int, float]:
        controller_index = self.params.centering_controller_index
        if self.state in {"STABILIZING", "BALANCED"}:
            controller_index = self.params.run_controller_index
        return controller_index, self._est_k

    def update(
        self,
        x_hat_0,
        innovation_0,
        x_hat_1,
        innovation_1,
        dt,
    ) -> tuple[int, float]:
        del innovation_0, innovation_1
        prev_state = self.state
        self._t_state += dt
        self._update_startup(x_hat_0, dt)
        if self.state == "STABILIZING":
            self._update_blend(x_hat_0, x_hat_1, dt)
            if self._est_k >= 1.0:
                self.state = "BALANCED"
                self._t_state = 0.0
        self._finish_update(prev_state)
        return self.active_output

    def reset(self):
        super().reset()
        self._t_state = 0.0
        self._reset_blend_progress()

    def _measurement_offset_latched(self) -> bool:
        return self.state in {"STABILIZING", "BALANCED"}

    def _on_upright_ready(self) -> None:
        self.state = "STABILIZING"
        self._t_state = 0.0
        self._reset_blend_progress()

    def _on_reset_to_acquisition(self) -> None:
        self._t_state = 0.0
        self._reset_blend_progress()

    def _reset_blend_progress(self) -> None:
        self._t_match = 0.0
        self._t_blend = 0.0
        self._est_k = 0.0
        self._blend_active = False

    def _update_blend(self, x_hat_0, x_hat_1, dt: float) -> None:
        if not self._blend_active:
            if not self._estimators_match(x_hat_0, x_hat_1):
                self._reset_blend_progress()
                return

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
