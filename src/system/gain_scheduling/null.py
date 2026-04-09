from src.shared import Measurement

from .base import GainScheduler


class NullGainScheduler(GainScheduler):
    def __init__(self, params):
        pass
    
    def apply(self, y_raw: Measurement) -> Measurement:
        return y_raw

    def map_angle(self, angle_rad: float) -> float:
        return float(angle_rad)
