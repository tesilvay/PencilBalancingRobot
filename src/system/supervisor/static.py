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

    def update(self, x_est, innovation, dt) -> tuple[int, int]:
        return self.params.controller_index, self.params.estimator_index
