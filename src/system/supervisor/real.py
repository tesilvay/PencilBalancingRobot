from __future__ import annotations

from dataclasses import dataclass, field

from src.shared import WorkspaceParams, default_workspace

from .base import RealServoSupervisorBase, RealStartupParams


@dataclass(kw_only=True)
class RealSupervisorParams(RealStartupParams):
    estimator_index: int
    workspace: WorkspaceParams = field(default_factory=default_workspace)


REAL_SUPERVISOR_PRESETS = {
    "default": {
        "centering_controller_index": 0,
        "run_controller_index": 1,
        "estimator_index": 0,
        "stable_threshold_deg": 3.0,
        "stable_hold_s": 2.0,
        "manual_step_m": 0.002,
    }
}


class RealSupervisor(RealServoSupervisorBase):
    def __init__(self, params: RealSupervisorParams):
        super().__init__(params)

    @property
    def active_indices(self) -> tuple[int, int]:
        controller_index = self.params.centering_controller_index
        if self.state == "BALANCED":
            controller_index = self.params.run_controller_index
        return controller_index, self.params.estimator_index

    def update(self, x_est, innovation, dt) -> tuple[int, int]:
        del innovation
        prev_state = self.state
        self._update_startup(x_est, dt)
        self._finish_update(prev_state)
        return self.active_indices

    def _measurement_offset_latched(self) -> bool:
        return self.state == "BALANCED"

    def _on_upright_ready(self) -> None:
        self.state = "BALANCED"
