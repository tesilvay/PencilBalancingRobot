from dataclasses import dataclass, field

from src.shared import TimingParams, default_timing


SCHEDULER_PRESETS = {
    "default": {
        "actuator_frequency": 250,
        "render_frequency":   30,
    }
}


@dataclass
class SchedulerParams:
    actuator_frequency: int
    render_frequency:   int
    timing:             TimingParams = field(default_factory=default_timing)


class Scheduler:
    def __init__(self, params: SchedulerParams):
        self.dt         = params.timing.dt
        self.actuator_dt = 1.0 / params.actuator_frequency
        self.render_dt  = 1.0 / params.render_frequency if params.render_frequency else None

        self.t             = 0.0
        self.next_actuator = 0.0
        self.next_render   = 0.0

    def tick(self):
        self.t += self.dt

    def should_actuate(self):
        if self.t >= self.next_actuator:
            self.next_actuator += self.actuator_dt
            return True
        return False

    def should_render(self):
        if self.render_dt is None:
            return False
        if self.t >= self.next_render:
            self.next_render += self.render_dt
            return True
        return False

    def reset(self):
        self.t             = 0.0
        self.next_actuator = 0.0
        self.next_render   = 0.0
