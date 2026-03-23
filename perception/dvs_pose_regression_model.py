"""
Runtime multivariate regression model for DVS pose estimation.

Implements the runtime estimator described in `docs/dvs_full_cal.md`:

    f(b1, s1, b2, s2) -> (X, Y, ax, ay)

where inputs are camnorm line parameters and outputs are pose in workspace
coordinates (meters, radians).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from core.sim_types import CameraPair, PoseMeasurement
#from perception.vision import get_measurements

def get_measurements(cams: CameraPair):
    b1 = cams.cam1.intercept
    s1 = cams.cam1.slope

    b2 = cams.cam2.intercept
    s2 = cams.cam2.slope
    
    return b1, s1, b2, s2


class DVSPoseRegressionModel:
    """
    Multivariate linear regression model for DVS pose estimation.

    Loads a JSON artifact of the form documented in `docs/dvs_full_cal.md` and
    provides an `estimate(cams)` method that returns a `PoseMeasurement`.
    """

    def __init__(
        self,
        input_order: list[str],
        input_mean: np.ndarray,
        input_std: np.ndarray,
        feature_order: list[str],
        output_order: list[str],
        W: np.ndarray,
        metadata: Dict[str, Any] | None = None,
    ):
        if len(input_order) != 4:
            raise ValueError(f"input_order must have 4 entries, got {input_order!r}")
        if W.shape != (4, 9):
            raise ValueError(f"W must have shape (4, 9), got {W.shape}")

        self.input_order = list(input_order)
        self.input_mean = np.asarray(input_mean, dtype=float).reshape(4)
        self.input_std = np.asarray(input_std, dtype=float).reshape(4)
        self.feature_order = list(feature_order)
        self.output_order = list(output_order)
        self.W = np.asarray(W, dtype=float)  # shape (4, 9)
        self.metadata = metadata or {}

        # Guard against division by zero
        self.input_std_safe = self.input_std.copy()
        self.input_std_safe[self.input_std_safe < 1e-8] = 1.0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "DVSPoseRegressionModel":
        """
        Load regression model from JSON file.
        """
        path = Path(path)
        with path.open("r") as f:
            data = json.load(f)

        if data.get("model_type") != "linear_regression":
            raise ValueError(f"Unexpected model_type in {path}: {data.get('model_type')!r}")

        input_order = data["input_order"]
        input_mean = np.array(data["input_mean"], dtype=float)
        input_std = np.array(data["input_std"], dtype=float)
        feature_order = data["feature_order"]
        output_order = data["output_order"]
        W = np.array(data["W"], dtype=float)

        metadata = {
            k: v
            for k, v in data.items()
            if k
            not in {
                "model_type",
                "input_order",
                "input_mean",
                "input_std",
                "feature_order",
                "output_order",
                "W",
            }
        }

        return cls(
            input_order=input_order,
            input_mean=input_mean,
            input_std=input_std,
            feature_order=feature_order,
            output_order=output_order,
            W=W,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Core math
    # ------------------------------------------------------------------
    def _standardize_inputs(self, cams: CameraPair) -> np.ndarray:
        """
        Standardize camnorm inputs using training mean/std.
        """
        # Map measured values into the order expected by the model
        b1, s1, b2, s2 = get_measurements(cams)
        # convert to dict
        values: Dict[str, float] = {"b1": float(b1), "s1": float(s1), "b2": float(b2), "s2": float(s2)}

        # convert to array
        x = np.array([values[name] for name in self.input_order], dtype=float)

        # standardize
        x_z = (x - self.input_mean) / self.input_std_safe
        return x_z  # shape (4,)

    def _build_feature_vector(self, x_z: np.ndarray) -> np.ndarray:
        """
        Build 9-dimensional feature vector in the order used during training:
            [1, b1_z, b2_z, s1_z, s2_z,
             b1_z*s1_z, b1_z*s2_z, b2_z*s1_z, b2_z*s2_z]
        """
        if x_z.shape[0] != 4:
            raise ValueError(f"Expected standardized input of length 4, got {x_z.shape}")

        # Recover named standardized variables according to input_order
        mapping = {name: x_z[i] for i, name in enumerate(self.input_order)}
        b1_z = mapping["b1"]
        s1_z = mapping["s1"]
        b2_z = mapping["b2"]
        s2_z = mapping["s2"]

        features = np.array(
            [
                1.0,
                b1_z,
                b2_z,
                s1_z,
                s2_z,
                b1_z * s1_z,
                b1_z * s2_z,
                b2_z * s1_z,
                b2_z * s2_z,
            ],
            dtype=float,
        )
        return features  # shape (9,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def estimate(self, cams: CameraPair) -> PoseMeasurement:
        """
        Estimate pose from a `CameraPair` of camnorm line observations.

        Uses:
            - standardization with training mean/std
            - polynomial feature expansion
            - linear map W @ features
        """
        x_z = self._standardize_inputs(cams)
        features = self._build_feature_vector(x_z)

        # outputs = [X, Y, ax, ay] in the units documented in the JSON
        y = self.W @ features  # shape (4,)
    
        return PoseMeasurement(X=float(y[0]), Y=float(y[1]), alpha_x=float(y[2]), alpha_y=float(y[3]))

