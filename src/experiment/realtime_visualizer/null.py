from src.shared import NullParams

from .base import RealtimeVisualizerBase, VizResult


class NullRealtimeVisualizer(RealtimeVisualizerBase):
    def __init__(self, params: NullParams):
        pass

    def render(self, measurement=None, command=None, **kwargs) -> VizResult:
        del measurement, command, kwargs
        return VizResult(quit=False)
