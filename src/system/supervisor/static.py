from dataclasses import dataclass

from .base import Supervisor


@dataclass
class StaticSupervisorParams:
    controller_index: int
    estimator_index: int


STATIC_SUPERVISOR_PRESETS = {
    "default": {
        "controller_index": 0,
        "estimator_index":  0,
    }
}


class StaticSupervisor(Supervisor):
    def __init__(self, params: StaticSupervisorParams):
        self.params = params
        self._last_transition: dict | None = None

    def update(self, x_est, innovation, dt) -> tuple[int, int]:
        self._last_transition = None
        return self.params.controller_index, self.params.estimator_index

    @property
    def last_transition(self) -> dict | None:
        return self._last_transition
