"""
Simple per-camera regression model for DVS pose estimation.

This is intentionally lightweight and manually calibratable:
  cam1: (slope_px, x_at_mask_px) -> (X, alpha_x)
  cam2: (slope_px, x_at_mask_px) -> (Y, alpha_y)

Mapping is affine (1D per output):
  X        = kx * x_at_mask + bx
  alpha_x  = ka * slope_px  + ba
  Y        = ky * x_at_mask + by
  alpha_y  = kb * slope_px  + bb
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from core.sim_types import CameraObservation, PoseMeasurement
from perception.dvs_algorithms import line_x_at_pixel_y


@dataclass(frozen=True)
class SimpleDVSCameraCalibration:
    """Affine maps for a single camera."""

    # Position axis (meters) from x_at_mask (pixels)
    k_pos: float
    b_pos: float

    # Tilt axis (radians) from slope_px (pixels/pixel)
    k_alpha: float
    b_alpha: float

    def estimate_axis(self, obs_px: CameraObservation, mask_y: int) -> tuple[float, float]:
        x_at_mask = float(line_x_at_pixel_y(obs_px, mask_y))
        s_px = float(obs_px.slope)
        pos = self.k_pos * x_at_mask + self.b_pos
        alpha = self.k_alpha * s_px + self.b_alpha
        return pos, alpha


@dataclass(frozen=True)
class SimpleDVSRegressionModel:
    """
    Stores independent calibrations for cam1 and cam2.

    Conventions:
    - cam1 estimates X and alpha_x
    - cam2 estimates Y and alpha_y
    """

    cam1: SimpleDVSCameraCalibration
    cam2: SimpleDVSCameraCalibration
    mask_y_cam1: int
    mask_y_cam2: int
    metadata: Dict[str, Any] | None = None

    def estimate_pose(self, obs1_px: CameraObservation, obs2_px: CameraObservation) -> PoseMeasurement:
        X, alpha_x = self.cam1.estimate_axis(obs1_px, mask_y=int(self.mask_y_cam1))
        Y, alpha_y = self.cam2.estimate_axis(obs2_px, mask_y=int(self.mask_y_cam2))
        return PoseMeasurement(X=float(X), Y=float(Y), alpha_x=float(alpha_x), alpha_y=float(alpha_y))

    # ------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "simple_dvs_regression_v1",
            "mask_y_cam1": int(self.mask_y_cam1),
            "mask_y_cam2": int(self.mask_y_cam2),
            "cam1": {
                "k_pos": float(self.cam1.k_pos),
                "b_pos": float(self.cam1.b_pos),
                "k_alpha": float(self.cam1.k_alpha),
                "b_alpha": float(self.cam1.b_alpha),
            },
            "cam2": {
                "k_pos": float(self.cam2.k_pos),
                "b_pos": float(self.cam2.b_pos),
                "k_alpha": float(self.cam2.k_alpha),
                "b_alpha": float(self.cam2.b_alpha),
            },
            "metadata": dict(self.metadata or {}),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SimpleDVSRegressionModel":
        path = Path(path)
        with path.open("r") as f:
            data = json.load(f)
        if data.get("model_type") != "simple_dvs_regression_v1":
            raise ValueError(f"Unexpected model_type in {path}: {data.get('model_type')!r}")

        cam1 = data["cam1"]
        cam2 = data["cam2"]

        return cls(
            cam1=SimpleDVSCameraCalibration(
                k_pos=float(cam1["k_pos"]),
                b_pos=float(cam1["b_pos"]),
                k_alpha=float(cam1["k_alpha"]),
                b_alpha=float(cam1["b_alpha"]),
            ),
            cam2=SimpleDVSCameraCalibration(
                k_pos=float(cam2["k_pos"]),
                b_pos=float(cam2["b_pos"]),
                k_alpha=float(cam2["k_alpha"]),
                b_alpha=float(cam2["b_alpha"]),
            ),
            mask_y_cam1=int(data["mask_y_cam1"]),
            mask_y_cam2=int(data["mask_y_cam2"]),
            metadata=data.get("metadata") or {},
        )


def fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Fit y ≈ k*x + b in least squares sense.
    Returns (k, b).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("x and y must have same length")
    if x.size < 2:
        raise ValueError("Need at least 2 samples to fit affine model")
    A = np.column_stack([x, np.ones_like(x)])
    (k, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(k), float(b)

