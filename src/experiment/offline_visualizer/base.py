from __future__ import annotations

from src.experiment.logger.logger import SimulationResult


class OfflineVisualizerBase:
    """Post-run visualization from logged trajectories."""

    def finalize(self, result: SimulationResult, *, dt: float) -> None:
        raise NotImplementedError
