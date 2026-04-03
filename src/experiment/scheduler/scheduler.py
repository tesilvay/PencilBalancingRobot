from dataclasses import dataclass

from src.shared import TimingParams

SCHEDULER_PRESETS = {
    "default": {
        "timing":             "default:default",
        "actuator_frequency": 250,
        "render_frequency":   30,
    }
}

@dataclass
class SchedulerParams:
    timing:             TimingParams
    actuator_frequency: int
    render_frequency:   int

class Scheduler:
    def __init__(self, dt, actuator_dt, render_dt=None):
        self.dt = dt
        self.actuator_dt = actuator_dt
        self.render_dt = render_dt

        self.t = 0
        self.next_actuator = 0
        self.next_render = 0

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
        self.t = 0
        self.next_actuator = 0
        self.next_render = 0
