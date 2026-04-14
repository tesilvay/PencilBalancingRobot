from dataclasses import dataclass, field

import numpy as np

from src.shared import WorkspaceParams, default_workspace

from .base import Supervisor


@dataclass
class DynamicSupervisorParams:
    radius_ramp_s: float = 5.0
    placing_time_s: float = 1.0
    min_radius: float = 0.0
    max_radius: float | None = None
    workspace: WorkspaceParams = field(default_factory=default_workspace)


DYNAMIC_SUPERVISOR_PRESETS = {
    "default": {
        "radius_ramp_s": 1.0,
        "placing_time_s": 1.0,
        "min_radius": 0.0,
        "max_radius": 4e-2,
    }
}


class DynamicSupervisor(Supervisor):
    """Two-state supervisor that only opens the PlacingPlant top radius."""

    def __init__(self, params: DynamicSupervisorParams):
        self.params = params
        self.workspace = params.workspace
        self.state = "ACQUISITION"
        self._t_placing = 0.0
        self._t_stabilizing = 0.0
        self._last_transition: dict | None = None
        self.min_radius = params.min_radius
        self.max_radius = params.workspace.safe_radius if None else params.max_radius
        self._top_radius = self.min_radius

    @property
    def active_output(self) -> tuple[int, float]:
        return 0, 0.0

    @property
    def is_offset_latched(self) -> bool:
        return self.state == "STABILIZATION"

    @property
    def is_ready_to_run(self) -> bool:
        return self.state == "STABILIZATION"

    @property
    def state_name(self) -> str:
        return self.state

    @property
    def last_transition(self) -> dict | None:
        return self._last_transition

    @property
    def top_radius(self) -> float:
        return self._top_radius

    def attach_runtime(self, actuator=None, workspace=None):
        del actuator
        if workspace is not None:
            self.workspace = workspace
        self._set_radius_for_state()

    def update(
        self,
        x_hat_0,
        innovation_0,
        x_hat_1,
        innovation_1,
        dt,
    ) -> tuple[int, float]:
        del x_hat_0, innovation_0, x_hat_1, innovation_1
        prev_state = self.state

        if self.state == "ACQUISITION":
            self._t_placing += max(float(dt), 0.0)
            if self._t_placing >= self._placing_time_s():
                self.state = "STABILIZATION"
                self._t_stabilizing = self._t_placing - self._placing_time_s()
        else:
            self._t_stabilizing += max(float(dt), 0.0)

        self._set_radius_for_state()
        self._last_transition = {
            "prev_state": prev_state,
            "new_state": self.state,
            "left_acquisition": prev_state == "ACQUISITION" and self.state != "ACQUISITION",
            "left_prestart": prev_state == "ACQUISITION" and self.state != "ACQUISITION",
        }
        return self.active_output

    def reset(self):
        self.state = "ACQUISITION"
        self._t_placing = 0.0
        self._t_stabilizing = 0.0
        self._top_radius = self.min_radius
        self._last_transition = None

    def _set_radius_for_state(self) -> None:
        if self.state == "ACQUISITION":
            self._top_radius = self.min_radius
            return

        ramp_s = max(float(self.params.radius_ramp_s), 0.0)
        alpha = 1.0 if ramp_s <= 0.0 else self._t_stabilizing / ramp_s
        alpha = float(np.clip(alpha, 0.0, 1.0))
        self._top_radius = (1.0 - alpha) * self.min_radius + alpha * self.max_radius

    def _placing_time_s(self) -> float:
        return max(float(self.params.placing_time_s), 0.0)


