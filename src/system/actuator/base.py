from abc import ABC, abstractmethod

import numpy as np


class Actuator(ABC):
    """Apply commands; :meth:`apply` returns logged joint geometry for the 3D visualizer."""

    def set_workspace_offset(self, dx: float, dy: float) -> None:
        del dx, dy

    def set_calibration_enabled(self, enabled: bool) -> None:
        del enabled

    def send(self, command) -> np.ndarray:
        """Compatibility wrapper for tools that want to send a command immediately."""
        return self.apply(command)

    def mech_joint_snapshot(self, command) -> np.ndarray:
        """(3, 2) joint points in mm, or NaN when no mechanism. Never touches hardware."""
        return np.full((3, 2), np.nan, dtype=float)

    @abstractmethod
    def apply(self, command) -> np.ndarray:
        """Send/actuate; return same joint snapshot as :meth:`mech_joint_snapshot` for logging."""

    @abstractmethod
    def reset(self) -> None: ...
