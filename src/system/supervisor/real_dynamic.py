from __future__ import annotations

from dataclasses import dataclass, field

from src.shared import WorkspaceParams, default_workspace

from .base import RealServoSupervisorBase, RealStartupParams


@dataclass(kw_only=True)
class RealDynamicSupervisorParams(RealStartupParams):
    acquisition_estimator_index: int
    run_estimator_index: int
    estimator_switch_delay_s: float
    workspace: WorkspaceParams = field(default_factory=default_workspace)


REAL_DYNAMIC_SUPERVISOR_PRESETS = {
    "default": {
        "centering_controller_index": 0,
        "run_controller_index": 1,
        "acquisition_estimator_index": 0,
        "run_estimator_index": 1,
        "stable_threshold_deg": 4.0,
        "stable_threshold_m": 20e-3,
        "stable_hold_s": 1.0,
        "estimator_switch_delay_s": 0.5,
        "manual_step_m": 0.002,
    }
}


class RealDynamicSupervisor(RealServoSupervisorBase):
    def __init__(self, params: RealDynamicSupervisorParams):
        super().__init__(params)
        self._t_state = 0.0

    @property
    def active_indices(self) -> tuple[int, int]:
        controller_index = self.params.centering_controller_index
        estimator_index = self.params.acquisition_estimator_index
        if self.state in {"STABILIZING", "BALANCED"}:
            controller_index = self.params.run_controller_index
        if self.state == "BALANCED":
            estimator_index = self.params.run_estimator_index
        return controller_index, estimator_index

    def update(self, x_est, innovation, dt) -> tuple[int, int]:
        del innovation
        prev_state = self.state
        self._t_state += dt
        self._update_startup(x_est, dt)
        if self.state == "STABILIZING" and self._t_state >= self.params.estimator_switch_delay_s:
            self.state = "BALANCED"
            self._t_state = 0.0
        self._finish_update(prev_state)
        return self.active_indices

    def reset(self):
        super().reset()
        self._t_state = 0.0

    def _measurement_offset_latched(self) -> bool:
        return self.state in {"STABILIZING", "BALANCED"}

    def _on_upright_ready(self) -> None:
        self.state = "STABILIZING"
        self._t_state = 0.0

    def _on_reset_to_acquisition(self) -> None:
        self._t_state = 0.0
