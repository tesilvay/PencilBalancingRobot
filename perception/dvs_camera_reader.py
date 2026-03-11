"""
DVS camera reader for DAVIS346 event cameras.
Wraps dv-processing for opening cameras and reading event batches.
"""

import threading
import numpy as np

# DAVIS346 resolution
DAVIS346_WIDTH = 346
DAVIS346_HEIGHT = 260


class DVSReader:
    """
    Wraps a single dv-processing DAVIS346 camera.
    Opens by serial (camera identifier from discovery) or first available if None.
    """

    def __init__(self, serial: str | None = None):
        """
        Open a DAVIS346 camera.
        Args:
            serial: Camera serial number (e.g. "00000499" for DAVIS346).
                   If None, opens the first available camera.
        """
        import dv_processing as dv

        if serial:
            self._capture = dv.io.camera.open(serial)
        else:
            self._capture = dv.io.camera.open()

        res = self._capture.getEventResolution()
        self._width = int(res[0])
        self._height = int(res[1])

        if self._width != DAVIS346_WIDTH or self._height != DAVIS346_HEIGHT:
            raise ValueError(
                f"Expected DAVIS346 resolution {DAVIS346_WIDTH}x{DAVIS346_HEIGHT}, "
                f"got {self._width}x{self._height}"
            )

    @property
    def resolution(self) -> tuple[int, int]:
        """(width, height) for CameraModel / PaperHoughLineAlgorithm."""
        return (self._width, self._height)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def get_event_batch(self) -> np.ndarray | None:
        """
        Get next event batch from the camera.
        Returns numpy array with 'x', 'y' fields (and optionally 't', 'p').
        Returns None if no events available.
        """
        events = self._capture.getNextEventBatch()
        if events is None:
            return None
        return events.numpy()

    def is_running(self) -> bool:
        """Whether the camera is still running."""
        return self._capture.isRunning()

    def close(self) -> None:
        """Release camera resources."""
        # dv-processing uses RAII; capture is released when object is destroyed.
        # Explicit close may not exist; we rely on garbage collection or
        # deleting the reference.
        self._capture = None
