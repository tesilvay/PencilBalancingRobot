"""
DVS grid-sweep calibration: load JSON from the calibration tool and apply
2D interpolation (b1, b2) -> (X, Y) at runtime.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import LinearNDInterpolator


class DVSGridCalibration:
    """
    Loads calibration data (list of {x, y, b1, b2}) and provides
    apply(b1, b2) -> (X, Y) via scattered 2D interpolation.
    """

    def __init__(self, points: list[dict[str, float]], metadata: dict[str, Any] | None = None):
        self.metadata = metadata or {}
        arr = np.array([[p["x"], p["y"], p["b1"], p["b2"]] for p in points], dtype=np.float64)
        self._x_cmd = arr[:, 0]
        self._y_cmd = arr[:, 1]
        self._b1 = arr[:, 2]
        self._b2 = arr[:, 3]
        self._interp_x = LinearNDInterpolator(
            (self._b1, self._b2), self._x_cmd, fill_value=np.nan
        )
        self._interp_y = LinearNDInterpolator(
            (self._b1, self._b2), self._y_cmd, fill_value=np.nan
        )

    def apply(self, b1: float, b2: float) -> tuple[float, float]:
        """
        Map normalized line intercepts (b1, b2) to workspace (X, Y).
        Returns (X, Y) in meters. If (b1, b2) is outside the calibration
        convex hull, returns (nan, nan); caller may fall back to geometric
        reconstruct.
        """
        x = float(self._interp_x(b1, b2))
        y = float(self._interp_y(b1, b2))
        return (x, y)

    @classmethod
    def load(cls, path: str | Path) -> "DVSGridCalibration":
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        if "points" in data:
            points = data["points"]
            metadata = {k: v for k, v in data.items() if k != "points"}
        elif isinstance(data, list):
            points = data
            metadata = {}
        else:
            raise ValueError("Calibration JSON must be a list of points or an object with 'points' key.")
        return cls(points=points, metadata=metadata)
