from dataclasses import dataclass

from .base import Supervisor


@dataclass
class StaticSupervisorParams:
    controller_key: str
    estimator_key:  str


STATIC_SUPERVISOR_PRESETS = {
    "default": {
        "controller_key": "smooth",
        "estimator_key":  "kalman",
    }
}


class StaticSupervisor(Supervisor):
    def __init__(self, params: StaticSupervisorParams):
        self.params = params

    def update(self, x_est, innovation, dt) -> tuple[str, str]:
        return self.params.controller_key, self.params.estimator_key
