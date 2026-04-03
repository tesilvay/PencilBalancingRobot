from dataclasses import dataclass, field
import time

from src.shared import TimingParams, default_timing
from .base import Pacing


@dataclass
class RealTimePacingParams:
    timing: TimingParams = field(default_factory=default_timing)


REALTIME_PACING_PRESETS = {
    "default": {}
}


class RealTimePacing(Pacing):
    def __init__(self, params: RealTimePacingParams):
        self.dt = params.timing.dt
        self.next_time = time.perf_counter()

    def pace(self):
        self.next_time += self.dt
        sleep = self.next_time - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
