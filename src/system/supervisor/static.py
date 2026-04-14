from dataclasses import dataclass

from src.shared import WorkspaceParams

from .base import Supervisor


@dataclass
class StaticSupervisorParams:
    top_radius: float | None = None


STATIC_SUPERVISOR_PRESETS = {
    "default": {
        "top_radius": None,
    }
}


class StaticSupervisor(Supervisor):
    """Supervisor that keeps the PlacingPlant top radius fully open."""

    def __init__(self, params: StaticSupervisorParams):
        self.params = params
        self.workspace: WorkspaceParams | None = None
        self._top_radius = 0.0
        self._last_transition: dict | None = None

    @property
    def active_output(self) -> tuple[int, float]:
        return 0, 0.0

    @property
    def is_offset_latched(self) -> bool:
        return True

    @property
    def state_name(self) -> str:
        return "STATIC"

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
        self._top_radius = self._max_radius()

    def update(self, x_hat_0, innovation_0, x_hat_1, innovation_1, dt) -> tuple[int, float]:
        del x_hat_0, innovation_0, x_hat_1, innovation_1, dt
        self._top_radius = self._max_radius()
        self._last_transition = None
        return self.active_output

    def reset(self):
        self._top_radius = self._max_radius()
        self._last_transition = None

    def _max_radius(self) -> float:
        if self.params.top_radius is not None:
            return max(float(self.params.top_radius), 0.0)
        if self.workspace is not None and self.workspace.safe_radius is not None:
            return max(float(self.workspace.safe_radius), 0.0)
        return 0.0
