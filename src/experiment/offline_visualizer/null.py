from src.shared import NullParams

from .base import OfflineVisualizerBase
from src.experiment.logger.logger import SimulationResult


class NullOfflineVisualizer(OfflineVisualizerBase):
    def __init__(self, params: NullParams):
        pass

    def finalize(self, result: SimulationResult, *, dt: float) -> None:
        del result, dt
        return None
