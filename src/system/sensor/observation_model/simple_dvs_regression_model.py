"""
Simple per-camera regression model for DVS y_meas estimation.

Two artifact kinds:

1) **Affine** (`model_type: simple_dvs_regression_v1`): per-camera affine maps from
   pixel `x_at_mask` and slope to position/tilt. Written by
   ``save_affine_v1_calibration`` after interactive calibration; loaded at runtime via
   ``load``.

2) **Bilinear** (`model_type: simple_dvs_regression_v2`): joint bilinear position model
   using both cameras' intercepts to correct for perspective cross-coupling.
   Position: px = a0 + a1*b1 + a2*b2, py = c0 + c1*b2 + c2*b1.
   Tilt uses same per-camera affine as v1. Written by ``save_bilinear_v2_calibration``.

3) **Calibration dataset** (`hardware/.../dvs_calibration_dataset.json`): staged b1/b2/s1/s2
   tables; runtime uses four 1D linear interpolations after converting camnorm lines to pixels.

Public estimate API: ``estimate(cams: CameraPair)`` — pixel-space observations per camera;
``x_at_mask`` is ``line_x_at_pixel_y`` at each camera's mask line. Returned y_meas is clamped:
tilts to ±10°, and (X, Y) radially to the workspace disk (``metadata.workspace_radius_m`` or
``safe_radius_m``, default 0.068 m, ref from ``workspace_x_ref_m`` / ``workspace_y_ref_m``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from src.shared import CameraObservation, CameraPair, Measurement, default_camera_params
from src.system.sensor.observation_model.camera_model import CameraModel
from src.system.sensor.algo.dvs_algorithms import line_x_at_pixel_y



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
class SimpleDVSV2PositionCalibration:
    """
    Bilinear position model using both cameras jointly.

    Corrects perspective cross-coupling: when the pencil is off-axis, each camera's
    intercept is distorted by the perpendicular offset. The cross-terms (a2, c2) capture
    this predictable shift so position is accurate across the full 2D workspace.

      px = a0 + a1*b1 + a2*b2
      py = c0 + c1*b2 + c2*b1
    """
    a0: float  # px intercept
    a1: float  # px: cam1 (own) coefficient
    a2: float  # px: cam2 (cross) coefficient
    c0: float  # py intercept
    c1: float  # py: cam2 (own) coefficient
    c2: float  # py: cam1 (cross) coefficient

    def estimate_position(self, b1: float, b2: float) -> tuple[float, float]:
        px = self.a0 + self.a1 * b1 + self.a2 * b2
        py = self.c0 + self.c1 * b2 + self.c2 * b1
        return px, py


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


# Post-estimate limits: tilts configurable, XY kept inside disk (see _clamp_measurement).
_SIMPLEDVS_MAX_TILT_RAD = float(np.deg2rad(10.0))
_DEFAULT_WORKSPACE_SAFE_RADIUS_M = 0.068
_DEFAULT_CAMERA_PARAMS = default_camera_params()


def _clamp_measurement(
    p: Measurement,
    metadata: Dict[str, Any] | None,
    max_tilt_rad: float,
) -> Measurement:
    """Clip tilts to configured bounds; project (X, Y) onto disk around ref with safe radius from metadata."""
    ax = float(np.clip(p.ax, -max_tilt_rad, max_tilt_rad))
    ay = float(np.clip(p.ay, -max_tilt_rad, max_tilt_rad))
    meta = dict(metadata or {})
    raw_r = meta.get("workspace_radius_m", meta.get("safe_radius_m", _DEFAULT_WORKSPACE_SAFE_RADIUS_M))
    safe_r = float(raw_r)
    x_ref = float(meta.get("workspace_x_ref_m", 0.0))
    y_ref = float(meta.get("workspace_y_ref_m", 0.0))
    X, Y = float(p.px), float(p.py)
    if safe_r > 0.0 and np.isfinite(safe_r):
        dx = X - x_ref
        dy = Y - y_ref
        dist = float(np.hypot(dx, dy))
        if dist > safe_r and dist > 0.0:
            s = safe_r / dist
            X = x_ref + dx * s
            Y = y_ref + dy * s
    return Measurement(px=X, py=Y, ax=ax, ay=ay)


@dataclass(frozen=True)
class SimpleDVSRegressionModel:
    """
    Affine calibration (cam1/cam2), bilinear v2, or dataset interpolation tables.

    Conventions:
    - cam1 estimates X and alpha_x
    - cam2 estimates Y and alpha_y

    Modes (mutually exclusive):
    - v1 affine: cam1 + cam2 set, pos_v2 is None
    - v2 bilinear: cam1 + cam2 set (tilt only), pos_v2 set
    - dataset: interp_X/Y/alpha_x/alpha_y set, cam1/cam2/pos_v2 are None
    """

    mask_y_cam1: int
    mask_y_cam2: int
    metadata: Dict[str, Any] | None = None
    max_tilt_rad: float = _SIMPLEDVS_MAX_TILT_RAD

    # Affine mode (v1) or tilt-only for v2
    cam1: SimpleDVSCameraCalibration | None = None
    cam2: SimpleDVSCameraCalibration | None = None

    # Bilinear position correction (v2); requires cam1+cam2 for tilt
    pos_v2: SimpleDVSV2PositionCalibration | None = None

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
        if self.pos_v2 is not None:
            if not has_affine:
                raise ValueError("v2 bilinear mode requires cam1 and cam2 (for tilt estimation)")
        elif has_affine == has_ds:
            raise ValueError("Set exactly one of: (cam1, cam2) affine pair or full dataset tables")

    def estimate(self, cams: CameraPair) -> Measurement:
        obs1_px = cams.cam1
        obs2_px = cams.cam2

        if self.pos_v2 is not None:
            b1 = float(line_x_at_pixel_y(obs1_px, float(self.mask_y_cam1)))
            b2 = float(line_x_at_pixel_y(obs2_px, float(self.mask_y_cam2)))
            X, Y = self.pos_v2.estimate_position(b1, b2)
            alpha_x = self.cam1.k_alpha * float(obs1_px.slope) + self.cam1.b_alpha  # type: ignore[union-attr]
            alpha_y = self.cam2.k_alpha * float(obs2_px.slope) + self.cam2.b_alpha  # type: ignore[union-attr]
            raw = Measurement(px=float(X), py=float(Y), ax=float(alpha_x), ay=float(alpha_y))
            return _clamp_measurement(raw, self.metadata, self.max_tilt_rad)

        if self.cam1 is not None and self.cam2 is not None:
            X, alpha_x = self.cam1.estimate_axis(obs1_px, mask_y=int(self.mask_y_cam1))
            Y, alpha_y = self.cam2.estimate_axis(obs2_px, mask_y=int(self.mask_y_cam2))
            raw = Measurement(px=float(X), py=float(Y), ax=float(alpha_x), ay=float(alpha_y))
            return _clamp_measurement(raw, self.metadata, self.max_tilt_rad)

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
        raw = Measurement(px=X, py=Y, ax=alpha_x, ay=alpha_y)
        return _clamp_measurement(raw, self.metadata, self.max_tilt_rad)

    # ------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        if self.cam1 is None or self.cam2 is None:
            raise TypeError("to_dict() is only supported for affine (v1/v2) models")
        cam1_dict = {
            "k_pos": float(self.cam1.k_pos),
            "b_pos": float(self.cam1.b_pos),
            "k_alpha": float(self.cam1.k_alpha),
            "b_alpha": float(self.cam1.b_alpha),
        }
        cam2_dict = {
            "k_pos": float(self.cam2.k_pos),
            "b_pos": float(self.cam2.b_pos),
            "k_alpha": float(self.cam2.k_alpha),
            "b_alpha": float(self.cam2.b_alpha),
        }
        if self.pos_v2 is not None:
            return {
                "model_type": "simple_dvs_regression_v2",
                "mask_y_cam1": int(self.mask_y_cam1),
                "mask_y_cam2": int(self.mask_y_cam2),
                "pos_v2": {
                    "a0": float(self.pos_v2.a0),
                    "a1": float(self.pos_v2.a1),
                    "a2": float(self.pos_v2.a2),
                    "c0": float(self.pos_v2.c0),
                    "c1": float(self.pos_v2.c1),
                    "c2": float(self.pos_v2.c2),
                },
                "cam1": cam1_dict,
                "cam2": cam2_dict,
                "metadata": dict(self.metadata or {}),
            }
        return {
            "model_type": "simple_dvs_regression_v1",
            "mask_y_cam1": int(self.mask_y_cam1),
            "mask_y_cam2": int(self.mask_y_cam2),
            "cam1": cam1_dict,
            "cam2": cam2_dict,
            "metadata": dict(self.metadata or {}),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(
        cls,
        path: str | Path,
        max_tilt_deg: float | None = None,
    ) -> "SimpleDVSRegressionModel":
        path = Path(path)
        with path.open("r") as f:
            data = json.load(f)

        max_tilt_rad = (
            _SIMPLEDVS_MAX_TILT_RAD
            if max_tilt_deg is None
            else float(np.deg2rad(max_tilt_deg))
        )

        if data.get("model_type") == "simple_dvs_regression_v1":
            cam1 = data["cam1"]
            cam2 = data["cam2"]
            return cls(
                mask_y_cam1=int(data["mask_y_cam1"]),
                mask_y_cam2=int(data["mask_y_cam2"]),
                metadata=data.get("metadata") or {},
                max_tilt_rad=max_tilt_rad,
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

        if data.get("model_type") == "simple_dvs_regression_v2":
            cam1 = data["cam1"]
            cam2 = data["cam2"]
            pv2 = data["pos_v2"]
            return cls(
                mask_y_cam1=int(data["mask_y_cam1"]),
                mask_y_cam2=int(data["mask_y_cam2"]),
                metadata=data.get("metadata") or {},
                max_tilt_rad=max_tilt_rad,
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
                pos_v2=SimpleDVSV2PositionCalibration(
                    a0=float(pv2["a0"]),
                    a1=float(pv2["a1"]),
                    a2=float(pv2["a2"]),
                    c0=float(pv2["c0"]),
                    c1=float(pv2["c1"]),
                    c2=float(pv2["c2"]),
                ),
            )

        if "stages" in data and all(k in data for k in ("b1", "b2", "s1", "s2")):
            return cls._from_calibration_dataset_dict(data, max_tilt_rad=max_tilt_rad)

        raise ValueError(
            f"Unrecognized JSON in {path}: expected simple_dvs_regression_v1/v2 or "
            f"calibration dataset with stages b1/b2/s1/s2"
        )

    @classmethod
    def _from_calibration_dataset_dict(
        cls,
        data: Dict[str, Any],
        max_tilt_rad: float = _SIMPLEDVS_MAX_TILT_RAD,
    ) -> "SimpleDVSRegressionModel":
        b1 = data["b1"]
        b2 = data["b2"]
        s1 = data["s1"]
        s2 = data["s2"]

        mask_y_cam1 = int(
            b1.get("mask_y_cam1", data.get("mask_y_cam1", _DEFAULT_CAMERA_PARAMS.y_mask_line_1))
        )
        mask_y_cam2 = int(
            b2.get("mask_y_cam2", data.get("mask_y_cam2", _DEFAULT_CAMERA_PARAMS.y_mask_line_2))
        )

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
            max_tilt_rad=max_tilt_rad,
            interp_X=interp_X,
            interp_Y=interp_Y,
            interp_alpha_x=interp_alpha_x,
            interp_alpha_y=interp_alpha_y,
        )


def default_affine_calibration_path() -> Path:
    """Absolute path next to this package (avoids cwd / permission surprises)."""
    return Path(__file__).resolve().parent / "calibration_files" / "simple_dvs_regression.json"


def _fit_affine_ls(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("x and y must have same length")
    if x.size < 2:
        raise ValueError("Need at least 2 samples to fit affine model")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("All sample values must be finite")
    A = np.column_stack([x, np.ones_like(x)])
    (k, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(k), float(b)


def save_affine_v1_calibration(
    path: str | Path,
    *,
    mask_y_cam1: int,
    mask_y_cam2: int,
    x_at_mask_px_cam1: Sequence[float],
    x_pos_m: Sequence[float],
    x_at_mask_px_cam2: Sequence[float],
    y_pos_m: Sequence[float],
    slope_px_cam1: Sequence[float],
    alpha_x_rad: Sequence[float],
    slope_px_cam2: Sequence[float],
    alpha_y_rad: Sequence[float],
    metadata: Dict[str, Any] | None = None,
) -> None:
    """
    Fit four least-squares affine maps and write ``simple_dvs_regression_v1`` JSON
    (same schema as ``SimpleDVSRegressionModel.to_dict`` / ``load``).
    """
    xa1 = np.asarray(x_at_mask_px_cam1, dtype=float).reshape(-1)
    xp = np.asarray(x_pos_m, dtype=float).reshape(-1)
    xa2 = np.asarray(x_at_mask_px_cam2, dtype=float).reshape(-1)
    yp = np.asarray(y_pos_m, dtype=float).reshape(-1)
    s1 = np.asarray(slope_px_cam1, dtype=float).reshape(-1)
    ax = np.asarray(alpha_x_rad, dtype=float).reshape(-1)
    s2 = np.asarray(slope_px_cam2, dtype=float).reshape(-1)
    ay = np.asarray(alpha_y_rad, dtype=float).reshape(-1)

    k1p, b1p = _fit_affine_ls(xa1, xp)
    k1a, b1a = _fit_affine_ls(s1, ax)
    k2p, b2p = _fit_affine_ls(xa2, yp)
    k2a, b2a = _fit_affine_ls(s2, ay)

    model = SimpleDVSRegressionModel(
        mask_y_cam1=int(mask_y_cam1),
        mask_y_cam2=int(mask_y_cam2),
        metadata=dict(metadata or {}),
        cam1=SimpleDVSCameraCalibration(k_pos=k1p, b_pos=b1p, k_alpha=k1a, b_alpha=b1a),
        cam2=SimpleDVSCameraCalibration(k_pos=k2p, b_pos=b2p, k_alpha=k2a, b_alpha=b2a),
    )
    model.save(path)


def save_bilinear_v2_calibration(
    path: str | Path,
    *,
    mask_y_cam1: int,
    mask_y_cam2: int,
    positions: Sequence[tuple[float, float]],
    b1_px: Sequence[float],
    b2_px: Sequence[float],
    slope_px_cam1: Sequence[float],
    alpha_x_rad: Sequence[float],
    slope_px_cam2: Sequence[float],
    alpha_y_rad: Sequence[float],
    metadata: Dict[str, Any] | None = None,
) -> None:
    """
    Fit bilinear position model and affine tilt maps, write ``simple_dvs_regression_v2`` JSON.

    Position model uses both cameras' intercepts jointly:
      px = a0 + a1*b1 + a2*b2
      py = c0 + c1*b2 + c2*b1

    Requires at least 3 grid points (for a well-determined 3-parameter fit); recommend 9+.
    """
    pos_arr = np.asarray(positions, dtype=float)
    b1_arr = np.asarray(b1_px, dtype=float).reshape(-1)
    b2_arr = np.asarray(b2_px, dtype=float).reshape(-1)

    n = len(b1_arr)
    if len(b2_arr) != n or len(pos_arr) != n:
        raise ValueError("positions, b1_px, b2_px must all have the same length")
    if n < 3:
        raise ValueError("Need at least 3 grid samples to fit bilinear position model")
    if not np.all(np.isfinite(b1_arr)) or not np.all(np.isfinite(b2_arr)):
        raise ValueError("All b1_px and b2_px values must be finite")

    px_m = pos_arr[:, 0]
    py_m = pos_arr[:, 1]

    # Design matrix columns: [1, b1, b2]
    # lstsq result layout: [intercept, coeff_of_b1, coeff_of_b2]
    A = np.column_stack([np.ones(n), b1_arr, b2_arr])
    px_coeffs, *_ = np.linalg.lstsq(A, px_m, rcond=None)
    py_coeffs, *_ = np.linalg.lstsq(A, py_m, rcond=None)

    # px = a0 + a1*b1 + a2*b2  (a1=own, a2=cross)
    a0, a1, a2 = float(px_coeffs[0]), float(px_coeffs[1]), float(px_coeffs[2])
    # py = c0 + c1*b2 + c2*b1  (c1=own, c2=cross)  → swap b1/b2 coeffs
    c0 = float(py_coeffs[0])
    c1 = float(py_coeffs[2])  # coeff of b2 (col 2) → c1 (cam2 own)
    c2 = float(py_coeffs[1])  # coeff of b1 (col 1) → c2 (cam1 cross)

    s1 = np.asarray(slope_px_cam1, dtype=float).reshape(-1)
    ax = np.asarray(alpha_x_rad, dtype=float).reshape(-1)
    s2 = np.asarray(slope_px_cam2, dtype=float).reshape(-1)
    ay = np.asarray(alpha_y_rad, dtype=float).reshape(-1)

    k1a, b1a = _fit_affine_ls(s1, ax)
    k2a, b2a = _fit_affine_ls(s2, ay)

    # k_pos/b_pos are set to the on-axis affine approximation (b2=0 → py ≈ c0 + c2*b1)
    # kept for reference but unused in v2 position estimation
    model = SimpleDVSRegressionModel(
        mask_y_cam1=int(mask_y_cam1),
        mask_y_cam2=int(mask_y_cam2),
        metadata=dict(metadata or {}),
        cam1=SimpleDVSCameraCalibration(k_pos=float(a1), b_pos=float(a0), k_alpha=k1a, b_alpha=b1a),
        cam2=SimpleDVSCameraCalibration(k_pos=float(c1), b_pos=float(c0), k_alpha=k2a, b_alpha=b2a),
        pos_v2=SimpleDVSV2PositionCalibration(
            a0=float(a0), a1=float(a1), a2=float(a2),
            c0=float(c0), c1=c1, c2=c2,
        ),
    )
    model.save(path)
