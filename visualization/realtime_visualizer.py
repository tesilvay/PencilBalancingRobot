"""Facade — re-exports from src.experiment.visualizer for backwards compatibility."""

try:
    from src.experiment.visualizer.one_dvs import OneDvsVisualizer, OneDvsVisualizerParams
except ImportError:
    class OneDvsVisualizerParams:
        """Stub — real implementation not present."""
        def __init__(self, **kwargs):
            pass

    class OneDvsVisualizer:
        """Stub — real implementation not present."""
        def __init__(self, *args, **kwargs):
            pass
