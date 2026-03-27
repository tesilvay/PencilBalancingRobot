"""
Simple per-camera regression model for DVS pose estimation.

Two artifact kinds:

1) **Affine** (`model_type: simple_dvs_regression_v1`): per-camera affine maps from
   pixel `x_at_mask` and slope to position/tilt (legacy `--full-regression`).

2) **Calibration dataset** (`hardware/.../dvs_calibration_dataset.json`): staged b1/b2/s1/s2
   tables; runtime uses four 1D linear interpolations after converting camnorm lines to pixels.

Public estimate API: ``estimate_pose(cams: CameraPair, cam: CameraModel)`` — observations
are **normalized** line parameters; the model converts with ``camnorm_to_pixel`` then evaluates
``x_at_mask`` via ``x = slope * y + intercept`` at each camera's mask line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from core.sim_types import CameraObservation, CameraPair, PoseMeasurement
from perception.camera_model import CameraModel
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


def _prepare_interp_table(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort by x and average y where x is duplicated (for stable np.interp).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("x and y must have same length")
    if x.size < 1:
        raise ValueError("Need at least one sample for interpolation table")
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]
    ux, inv = np.unique(x, return_inverse=True)
    if len(ux) == len(x):
        return ux, y
    sums = np.bincount(inv, weights=y)
    counts = np.bincount(inv)
    return ux, sums / np.maximum(counts, 1)


def _interp1d(xq: float, xp: np.ndarray, fp: np.ndarray) -> float:
    return float(np.interp(float(xq), xp, fp))


@dataclass(frozen=True)
class SimpleDVSRegressionModel:
    """
    Affine calibration (cam1/cam2) **or** dataset interpolation tables — mutually exclusive.

    Conventions:
    - cam1 estimates X and alpha_x
    - cam2 estimates Y and alpha_y
    """

    mask_y_cam1: int
    mask_y_cam2: int
    metadata: Dict[str, Any] | None = None

    # Affine mode (legacy v1)
    cam1: SimpleDVSCameraCalibration | None = None
    cam2: SimpleDVSCameraCalibration | None = None

    # Dataset mode: each is (xp, fp) sorted for np.interp
    interp_X: tuple[np.ndarray, np.ndarray] | None = None  # x_at_mask cam1 -> X [m]
    interp_Y: tuple[np.ndarray, np.ndarray] | None = None  # x_at_mask cam2 -> Y [m]
    interp_alpha_x: tuple[np.ndarray, np.ndarray] | None = None  # s1_px -> alpha_x [rad]
    interp_alpha_y: tuple[np.ndarray, np.ndarray] | None = None  # s2_px -> alpha_y [rad]

    def __post_init__(self) -> None:
        has_affine = self.cam1 is not None and self.cam2 is not None
        has_ds = (
            self.interp_X is not None
            and self.interp_Y is not None
            and self.interp_alpha_x is not None
            and self.interp_alpha_y is not None
        )
        if has_affine == has_ds:
            raise ValueError("Set exactly one of: (cam1, cam2) affine pair or full dataset tables")

    def estimate_pose(self, cams: CameraPair, cam: CameraModel) -> PoseMeasurement:
        obs1_px = cam.camnorm_to_pixel(cams.cam1)
        obs2_px = cam.camnorm_to_pixel(cams.cam2)
        if self.cam1 is not None and self.cam2 is not None:
            X, alpha_x = self.cam1.estimate_axis(obs1_px, mask_y=int(self.mask_y_cam1))
            Y, alpha_y = self.cam2.estimate_axis(obs2_px, mask_y=int(self.mask_y_cam2))
            return PoseMeasurement(X=float(X), Y=float(Y), alpha_x=float(alpha_x), alpha_y=float(alpha_y))

        x1 = float(line_x_at_pixel_y(obs1_px, float(self.mask_y_cam1)))
        x2 = float(line_x_at_pixel_y(obs2_px, float(self.mask_y_cam2)))
        s1 = float(obs1_px.slope)
        s2 = float(obs2_px.slope)

        xp_x, fp_x = self.interp_X  # type: ignore[misc]
        xp_y, fp_y = self.interp_Y  # type: ignore[misc]
        xp_ax, fp_ax = self.interp_alpha_x  # type: ignore[misc]
        xp_ay, fp_ay = self.interp_alpha_y  # type: ignore[misc]

        X = _interp1d(x1, xp_x, fp_x)
        Y = _interp1d(x2, xp_y, fp_y)
        alpha_x = _interp1d(s1, xp_ax, fp_ax)
        alpha_y = _interp1d(s2, xp_ay, fp_ay)
        return PoseMeasurement(X=X, Y=Y, alpha_x=alpha_x, alpha_y=alpha_y)

    # ------------------------------------------------------------
    # Serialization (affine only)
    # ------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        if self.cam1 is None or self.cam2 is None:
            raise TypeError("to_dict() is only supported for affine (v1) models")
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

        if data.get("model_type") == "simple_dvs_regression_v1":
            cam1 = data["cam1"]
            cam2 = data["cam2"]
            return cls(
                mask_y_cam1=int(data["mask_y_cam1"]),
                mask_y_cam2=int(data["mask_y_cam2"]),
                metadata=data.get("metadata") or {},
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
            )

        if "stages" in data and all(k in data for k in ("b1", "b2", "s1", "s2")):
            return cls._from_calibration_dataset_dict(data)

        raise ValueError(
            f"Unrecognized JSON in {path}: expected simple_dvs_regression_v1 or "
            f"calibration dataset with stages b1/b2/s1/s2"
        )

    @classmethod
    def _from_calibration_dataset_dict(cls, data: Dict[str, Any]) -> "SimpleDVSRegressionModel":
        b1 = data["b1"]
        b2 = data["b2"]
        s1 = data["s1"]
        s2 = data["s2"]

        mask_y_cam1 = int(b1.get("mask_y_cam1", data.get("mask_y_cam1", 160)))
        mask_y_cam2 = int(b2.get("mask_y_cam2", data.get("mask_y_cam2", 190)))

        b1_samples = b1["samples"]
        b2_samples = b2["samples"]
        s1_samples = s1["samples"]
        s2_samples = s2["samples"]

        x_at_b1 = np.array([float(s["x_at_mask_px"]) for s in b1_samples], dtype=float)
        x_pos_m = np.array([float(s["x_pos_m"]) for s in b1_samples], dtype=float)
        x_at_b2 = np.array([float(s["x_at_mask_px"]) for s in b2_samples], dtype=float)
        y_pos_m = np.array([float(s["y_pos_m"]) for s in b2_samples], dtype=float)

        s1_px = np.array([float(s["s1_px"]) for s in s1_samples], dtype=float)
        alpha_x_rad = np.array([float(s["alpha_x_rad"]) for s in s1_samples], dtype=float)
        s2_px = np.array([float(s["s2_px"]) for s in s2_samples], dtype=float)
        alpha_y_rad = np.array([float(s["alpha_y_rad"]) for s in s2_samples], dtype=float)

        interp_X = _prepare_interp_table(x_at_b1, x_pos_m)
        interp_Y = _prepare_interp_table(x_at_b2, y_pos_m)
        interp_alpha_x = _prepare_interp_table(s1_px, alpha_x_rad)
        interp_alpha_y = _prepare_interp_table(s2_px, alpha_y_rad)

        meta = dict(data.get("metadata") or {})
        meta.setdefault("source", "dvs_calibration_dataset")

        return cls(
            mask_y_cam1=mask_y_cam1,
            mask_y_cam2=mask_y_cam2,
            metadata=meta,
            interp_X=interp_X,
            interp_Y=interp_Y,
            interp_alpha_x=interp_alpha_x,
            interp_alpha_y=interp_alpha_y,
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
