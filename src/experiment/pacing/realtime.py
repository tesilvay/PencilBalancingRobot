from dataclasses import dataclass
import time

from src.shared import TimingParams
from .base import Pacing


@dataclass
class RealTimePacingParams:
    timing: TimingParams


REALTIME_PACING_PRESETS = {
    "default": {
        "timing": "default:default",
    }
}


class RealTimePacing(Pacing):
    def __init__(self, dt):
        self.dt = dt
        self.next_time = time.perf_counter()

    def pace(self):
        self.next_time += self.dt
        sleep = self.next_time - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
