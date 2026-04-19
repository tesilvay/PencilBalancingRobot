"""
Simple per-camera regression model for DVS y_meas estimation.

Three artifact kinds:

1) **Affine** (`model_type: simple_dvs_regression_v1`): per-camera affine maps from
   pixel `x_at_mask` and slope to position/tilt. Written by
   ``save_affine_v1_calibration`` after interactive calibration; loaded at runtime via
   ``load``.

2) **Bilinear** (`model_type: simple_dvs_regression_v2`): joint bilinear position model
   using both cameras' intercepts to correct for perspective cross-coupling.
   Position: px = a0 + a1*b1 + a2*b2, py = c0 + c1*b2 + c2*b1.
   Tilt uses same per-camera affine as v1. Written by ``save_bilinear_v2_calibration``.

3) **TPS** (`version: v3_tps`): thin-plate-spline non-parametric calibration.
   Position: 2D RBF over dense grid → exact interpolation at training samples.
   Tilt: 3D RBF over (px_m, py_m, alpha_rad) → position-dependent tilt.
   Written by ``save_tps_v3_calibration``. Highest priority when present.

Public estimate API: ``estimate(cams: CameraPair)`` — pixel-space observations per camera;
``x_at_mask`` is ``line_x_at_pixel_y`` at each camera's mask line. Returned y_meas is clamped:
tilts to ±10°, and (X, Y) radially to the workspace disk (``metadata.workspace_radius_m`` or
``safe_radius_m``, default 0.068 m, ref from ``workspace_x_ref_m`` / ``workspace_y_ref_m``).

Dispatch priority: v3_tps > v2_bilinear > v1_affine > dataset.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import brentq

from src.shared import CameraObservation, CameraPair, Measurement, default_camera_params
from src.system.sensor.observation_model.camera_model import CameraModel
from src.system.sensor.algo.dvs_algorithms import line_x_at_pixel_y


@dataclass(frozen=True)
class SimpleDVSCameraCalibration:
    """Affine position map + piecewise-linear tilt map for a single camera."""

    # Position axis (meters) from x_at_mask (pixels)
    k_pos: float
    b_pos: float

    # Tilt axis (radians) from slope_px (pixels/pixel) — piecewise linear interp table.
    # tilt_interp: (slopes_px_sorted, alphas_rad) for np.interp
    tilt_interp: tuple

    def estimate_axis(self, obs_px: CameraObservation, mask_y: int) -> tuple[float, float]:
        x_at_mask = float(line_x_at_pixel_y(obs_px, mask_y))
        s_px = float(obs_px.slope)
        pos = self.k_pos * x_at_mask + self.b_pos
        alpha = _interp1d(s_px, self.tilt_interp[0], self.tilt_interp[1])
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


@dataclass
class TPSCalibrationV3:
    """
    Thin-plate-spline calibration.
    Stores the raw sample data; RBF interpolators are built on load.

    Position:  (px_m, py_m) -> (b1_px, b2_px)   — 2D forward TPS
    Tilt cam1: (px_m, py_m, alpha_x_rad) -> s_px_cam1  — 3D → 1D TPS
    Tilt cam2: (px_m, py_m, alpha_y_rad) -> s_px_cam2  — 3D → 1D TPS

    Flat tilt lists are position-major / angle-inner:
      flat_idx = pi * len(alphas_rad) + ai
    """
    mask_y_cam1: int
    mask_y_cam2: int
    # Position (dense grid)
    positions_m: list[tuple[float, float]]   # length N_pos
    b1_px: list[float]                       # length N_pos
    b2_px: list[float]                       # length N_pos
    # Tilt (sparse 3D grid)
    tilt_positions_m: list[tuple[float, float]]  # length N_tilt_pos
    tilt_alphas_rad: list[float]                 # length N_tilt_ang
    # Flattened (positions outer, angles inner), length N_tilt_pos * N_tilt_ang
    tilt_s_px_cam1: list[float]
    tilt_s_px_cam2: list[float]


def _prepare_interp_table(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by x and average y where x is duplicated (for stable np.interp)."""
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


def _load_tilt_interp(cam_dict: dict) -> tuple:
    if "tilt_slopes_px" in cam_dict:
        xp = np.array(cam_dict["tilt_slopes_px"], dtype=float)
        fp = np.array(cam_dict["tilt_alphas_rad"], dtype=float)
        return _prepare_interp_table(xp, fp)
    # Backward compat: reconstruct 2-point table from old affine k_alpha/b_alpha
    k = float(cam_dict["k_alpha"])
    b = float(cam_dict["b_alpha"])
    xp = np.array([-0.2, 0.2], dtype=float)
    return _prepare_interp_table(xp, k * xp + b)


_SIMPLEDVS_MAX_TILT_RAD = float(np.deg2rad(25.0))
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
    Calibration model dispatching v3_tps > v2_bilinear > v1_affine > dataset.

    Conventions:
    - cam1 estimates X and alpha_x
    - cam2 estimates Y and alpha_y

    Modes (by priority, highest first):
    - v3 TPS: pos_v3 set; cam1/cam2/pos_v2 not required
    - v2 bilinear: cam1 + cam2 set (tilt only), pos_v2 set
    - v1 affine: cam1 + cam2 set, pos_v2 is None
    - dataset: interp_X/Y/alpha_x/alpha_y set, cam1/cam2/pos_v2 are None
    """

    mask_y_cam1: int
    mask_y_cam2: int
    metadata: Dict[str, Any] | None = None
    max_tilt_rad: float = _SIMPLEDVS_MAX_TILT_RAD

    # v3 TPS (highest priority)
    pos_v3: TPSCalibrationV3 | None = None

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
        if self.pos_v3 is not None:
            self._build_v3_interpolators()
            return

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

    def _build_v3_interpolators(self) -> None:
        v3 = self.pos_v3
        assert v3 is not None

        xy = np.array(v3.positions_m, dtype=float)  # (N_pos, 2)
        rbf_b1 = RBFInterpolator(xy, np.array(v3.b1_px, dtype=float), kernel="thin_plate_spline")
        rbf_b2 = RBFInterpolator(xy, np.array(v3.b2_px, dtype=float), kernel="thin_plate_spline")

        N_p = len(v3.tilt_positions_m)
        N_a = len(v3.tilt_alphas_rad)
        pts = np.empty((N_p * N_a, 3), dtype=float)
        for pi, (px, py) in enumerate(v3.tilt_positions_m):
            for ai, a in enumerate(v3.tilt_alphas_rad):
                pts[pi * N_a + ai] = (px, py, a)
        rbf_s1 = RBFInterpolator(pts, np.array(v3.tilt_s_px_cam1, dtype=float), kernel="thin_plate_spline")
        rbf_s2 = RBFInterpolator(pts, np.array(v3.tilt_s_px_cam2, dtype=float), kernel="thin_plate_spline")

        # bypass frozen restriction to cache the interpolators
        object.__setattr__(self, "_rbf_b1", rbf_b1)
        object.__setattr__(self, "_rbf_b2", rbf_b2)
        object.__setattr__(self, "_rbf_s1", rbf_s1)
        object.__setattr__(self, "_rbf_s2", rbf_s2)

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def pixel_to_world(self, b1: float, b2: float) -> tuple[float, float]:
        """Invert the position calibration. v3: Newton on TPS. v2: 2×2 linear solve. v1: per-axis affine."""
        if self.pos_v3 is not None:
            return self._pixel_to_world_v3(b1, b2)
        if self.pos_v2 is not None:
            return self.pos_v2.estimate_position(b1, b2)
        if self.cam1 is not None and self.cam2 is not None:
            x = self.cam1.k_pos * b1 + self.cam1.b_pos
            y = self.cam2.k_pos * b2 + self.cam2.b_pos
            return x, y
        raise ValueError("No position calibration available")

    def slope_to_alpha_x(self, px: float, py: float, slope_cam1: float) -> float:
        """Invert cam1 tilt model. v3: brentq on 3D TPS. v1/v2: 1D interp."""
        if self.pos_v3 is not None:
            return self._slope_to_alpha_v3(px, py, slope_cam1, cam=1)
        if self.cam1 is not None:
            return _interp1d(slope_cam1, self.cam1.tilt_interp[0], self.cam1.tilt_interp[1])
        raise ValueError("No cam1 tilt calibration available")

    def slope_to_alpha_y(self, px: float, py: float, slope_cam2: float) -> float:
        """Invert cam2 tilt model. v3: brentq on 3D TPS. v1/v2: 1D interp."""
        if self.pos_v3 is not None:
            return self._slope_to_alpha_v3(px, py, slope_cam2, cam=2)
        if self.cam2 is not None:
            return _interp1d(slope_cam2, self.cam2.tilt_interp[0], self.cam2.tilt_interp[1])
        raise ValueError("No cam2 tilt calibration available")

    def _pixel_to_world_v3(self, b1_obs: float, b2_obs: float) -> tuple[float, float]:
        v3 = self.pos_v3
        assert v3 is not None
        rbf_b1: RBFInterpolator = self._rbf_b1  # type: ignore[attr-defined]
        rbf_b2: RBFInterpolator = self._rbf_b2  # type: ignore[attr-defined]

        # Warm-start: nearest training sample by pixel distance
        b1_arr = np.array(v3.b1_px, dtype=float)
        b2_arr = np.array(v3.b2_px, dtype=float)
        dists = (b1_arr - b1_obs) ** 2 + (b2_arr - b2_obs) ** 2
        seed_idx = int(np.argmin(dists))
        p = np.array(v3.positions_m[seed_idx], dtype=float)

        # Newton iteration: solve rbf_b1(p) = b1_obs, rbf_b2(p) = b2_obs
        h = 1e-4
        for _ in range(10):
            p_in = p.reshape(1, 2)
            b1_p = float(rbf_b1(p_in)[0])
            b2_p = float(rbf_b2(p_in)[0])
            F = np.array([b1_p - b1_obs, b2_p - b2_obs])
            if float(np.linalg.norm(F)) < 1e-3:
                break
            b1_dx = float(rbf_b1(p_in + np.array([[h, 0.0]]))[0])
            b1_dy = float(rbf_b1(p_in + np.array([[0.0, h]]))[0])
            b2_dx = float(rbf_b2(p_in + np.array([[h, 0.0]]))[0])
            b2_dy = float(rbf_b2(p_in + np.array([[0.0, h]]))[0])
            J = np.array([
                [(b1_dx - b1_p) / h, (b1_dy - b1_p) / h],
                [(b2_dx - b2_p) / h, (b2_dy - b2_p) / h],
            ])
            try:
                dp = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                break
            p = p + dp

        # Clamp to workspace radius
        meta = dict(self.metadata or {})
        safe_r = float(meta.get("workspace_radius_m", meta.get("safe_radius_m", _DEFAULT_WORKSPACE_SAFE_RADIUS_M)))
        x_ref = float(meta.get("workspace_x_ref_m", 0.0))
        y_ref = float(meta.get("workspace_y_ref_m", 0.0))
        dx, dy = p[0] - x_ref, p[1] - y_ref
        dist = float(np.hypot(dx, dy))
        if safe_r > 0.0 and dist > safe_r:
            warnings.warn(
                f"TPS inverse clamped: converged {dist*100:.1f}cm from center > safe_r={safe_r*100:.1f}cm",
                stacklevel=3,
            )
            s = safe_r / dist
            p[0] = x_ref + dx * s
            p[1] = y_ref + dy * s

        return float(p[0]), float(p[1])

    def _slope_to_alpha_v3(self, px: float, py: float, slope_obs: float, cam: int) -> float:
        v3 = self.pos_v3
        assert v3 is not None
        rbf: RBFInterpolator = self._rbf_s1 if cam == 1 else self._rbf_s2  # type: ignore[attr-defined]

        a_lo = min(v3.tilt_alphas_rad)
        a_hi = max(v3.tilt_alphas_rad)

        def f(alpha: float) -> float:
            return float(rbf(np.array([[px, py, alpha]]))[0]) - slope_obs

        f_lo = f(a_lo)
        f_hi = f(a_hi)

        if f_lo * f_hi > 0:
            # slope outside calibrated range — extrapolate linearly using average dS/dα
            # over the calibrated band, then clamp to max_tilt_rad.
            # f(α) = rbf(α) - slope_obs, so f_hi - f_lo = s(a_hi) - s(a_lo).
            df_da = (f_hi - f_lo) / (a_hi - a_lo)
            if abs(df_da) < 1e-12:
                return a_lo if abs(f_lo) < abs(f_hi) else a_hi
            anchor_a = a_lo if abs(f_lo) < abs(f_hi) else a_hi
            anchor_f = f_lo if abs(f_lo) < abs(f_hi) else f_hi
            alpha_extrap = anchor_a - anchor_f / df_da
            return float(np.clip(alpha_extrap, -self.max_tilt_rad, self.max_tilt_rad))

        return float(brentq(f, a_lo, a_hi, xtol=1e-6))

    def estimate(self, cams: CameraPair) -> Measurement:
        obs1_px = cams.cam1
        obs2_px = cams.cam2

        if self.pos_v3 is not None:
            b1 = float(line_x_at_pixel_y(obs1_px, float(self.mask_y_cam1)))
            b2 = float(line_x_at_pixel_y(obs2_px, float(self.mask_y_cam2)))
            X, Y = self.pixel_to_world(b1, b2)
            alpha_x = self.slope_to_alpha_x(X, Y, float(obs1_px.slope))
            alpha_y = self.slope_to_alpha_y(X, Y, float(obs2_px.slope))
            raw = Measurement(px=float(X), py=float(Y), ax=float(alpha_x), ay=float(alpha_y))
            return _clamp_measurement(raw, self.metadata, self.max_tilt_rad)

        if self.pos_v2 is not None:
            b1 = float(line_x_at_pixel_y(obs1_px, float(self.mask_y_cam1)))
            b2 = float(line_x_at_pixel_y(obs2_px, float(self.mask_y_cam2)))
            X, Y = self.pos_v2.estimate_position(b1, b2)
            alpha_x = _interp1d(float(obs1_px.slope), self.cam1.tilt_interp[0], self.cam1.tilt_interp[1])  # type: ignore[union-attr]
            alpha_y = _interp1d(float(obs2_px.slope), self.cam2.tilt_interp[0], self.cam2.tilt_interp[1])  # type: ignore[union-attr]
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

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        if self.cam1 is None or self.cam2 is None:
            raise TypeError("to_dict() is only supported for affine (v1/v2) models")
        cam1_dict = {
            "k_pos": float(self.cam1.k_pos),
            "b_pos": float(self.cam1.b_pos),
            "tilt_slopes_px": self.cam1.tilt_interp[0].tolist(),
            "tilt_alphas_rad": self.cam1.tilt_interp[1].tolist(),
        }
        cam2_dict = {
            "k_pos": float(self.cam2.k_pos),
            "b_pos": float(self.cam2.b_pos),
            "tilt_slopes_px": self.cam2.tilt_interp[0].tolist(),
            "tilt_alphas_rad": self.cam2.tilt_interp[1].tolist(),
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

        if data.get("version") == "v3_tps":
            v3 = TPSCalibrationV3(
                mask_y_cam1=int(data["mask_y_cam1"]),
                mask_y_cam2=int(data["mask_y_cam2"]),
                positions_m=[tuple(p) for p in data["positions_m"]],  # type: ignore[misc]
                b1_px=[float(v) for v in data["b1_px"]],
                b2_px=[float(v) for v in data["b2_px"]],
                tilt_positions_m=[tuple(p) for p in data["tilt_positions_m"]],  # type: ignore[misc]
                tilt_alphas_rad=[float(v) for v in data["tilt_alphas_rad"]],
                tilt_s_px_cam1=[float(v) for v in data["tilt_s_px_cam1"]],
                tilt_s_px_cam2=[float(v) for v in data["tilt_s_px_cam2"]],
            )
            return cls(
                mask_y_cam1=v3.mask_y_cam1,
                mask_y_cam2=v3.mask_y_cam2,
                metadata=data.get("metadata") or {},
                max_tilt_rad=max_tilt_rad,
                pos_v3=v3,
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
                    tilt_interp=_load_tilt_interp(cam1),
                ),
                cam2=SimpleDVSCameraCalibration(
                    k_pos=float(cam2["k_pos"]),
                    b_pos=float(cam2["b_pos"]),
                    tilt_interp=_load_tilt_interp(cam2),
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
                    tilt_interp=_load_tilt_interp(cam1),
                ),
                cam2=SimpleDVSCameraCalibration(
                    k_pos=float(cam2["k_pos"]),
                    b_pos=float(cam2["b_pos"]),
                    tilt_interp=_load_tilt_interp(cam2),
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
            f"Unrecognized JSON in {path}: expected v3_tps, simple_dvs_regression_v1/v2, "
            f"or calibration dataset with stages b1/b2/s1/s2"
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
    Fit four least-squares affine maps and write ``simple_dvs_regression_v1`` JSON.
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
    k2p, b2p = _fit_affine_ls(xa2, yp)
    tilt_interp1 = _prepare_interp_table(s1, ax)
    tilt_interp2 = _prepare_interp_table(s2, ay)

    model = SimpleDVSRegressionModel(
        mask_y_cam1=int(mask_y_cam1),
        mask_y_cam2=int(mask_y_cam2),
        metadata=dict(metadata or {}),
        cam1=SimpleDVSCameraCalibration(k_pos=k1p, b_pos=b1p, tilt_interp=tilt_interp1),
        cam2=SimpleDVSCameraCalibration(k_pos=k2p, b_pos=b2p, tilt_interp=tilt_interp2),
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

    A = np.column_stack([np.ones(n), b1_arr, b2_arr])
    px_coeffs, *_ = np.linalg.lstsq(A, px_m, rcond=None)
    py_coeffs, *_ = np.linalg.lstsq(A, py_m, rcond=None)

    a0, a1, a2 = float(px_coeffs[0]), float(px_coeffs[1]), float(px_coeffs[2])
    c0 = float(py_coeffs[0])
    c1 = float(py_coeffs[2])  # coeff of b2 → c1 (cam2 own)
    c2 = float(py_coeffs[1])  # coeff of b1 → c2 (cam1 cross)

    s1 = np.asarray(slope_px_cam1, dtype=float).reshape(-1)
    ax = np.asarray(alpha_x_rad, dtype=float).reshape(-1)
    s2 = np.asarray(slope_px_cam2, dtype=float).reshape(-1)
    ay = np.asarray(alpha_y_rad, dtype=float).reshape(-1)

    tilt_interp1 = _prepare_interp_table(s1, ax)
    tilt_interp2 = _prepare_interp_table(s2, ay)

    model = SimpleDVSRegressionModel(
        mask_y_cam1=int(mask_y_cam1),
        mask_y_cam2=int(mask_y_cam2),
        metadata=dict(metadata or {}),
        cam1=SimpleDVSCameraCalibration(k_pos=float(a1), b_pos=float(a0), tilt_interp=tilt_interp1),
        cam2=SimpleDVSCameraCalibration(k_pos=float(c1), b_pos=float(c0), tilt_interp=tilt_interp2),
        pos_v2=SimpleDVSV2PositionCalibration(
            a0=float(a0), a1=float(a1), a2=float(a2),
            c0=float(c0), c1=c1, c2=c2,
        ),
    )
    model.save(path)


def save_tps_v3_calibration(
    path: str | Path,
    *,
    mask_y_cam1: int,
    mask_y_cam2: int,
    positions: list[tuple[float, float]],
    b1_px: list[float],
    b2_px: list[float],
    tilt_positions: list[tuple[float, float]],
    tilt_alphas_rad: list[float],
    tilt_s_px_cam1: list[float],
    tilt_s_px_cam2: list[float],
    metadata: dict | None = None,
) -> None:
    """Write v3_tps calibration JSON. Raw samples only; interpolators are rebuilt on load."""
    n = len(positions)
    if len(b1_px) != n or len(b2_px) != n:
        raise ValueError("positions, b1_px, b2_px must all have the same length")
    n_flat = len(tilt_positions) * len(tilt_alphas_rad)
    if len(tilt_s_px_cam1) != n_flat or len(tilt_s_px_cam2) != n_flat:
        raise ValueError(
            f"tilt_s_px_cam1/cam2 must have length n_tilt_pos * n_tilt_ang = {n_flat}"
        )

    data: dict = {
        "version": "v3_tps",
        "mask_y_cam1": int(mask_y_cam1),
        "mask_y_cam2": int(mask_y_cam2),
        "positions_m": [list(p) for p in positions],
        "b1_px": [float(v) for v in b1_px],
        "b2_px": [float(v) for v in b2_px],
        "tilt_positions_m": [list(p) for p in tilt_positions],
        "tilt_alphas_rad": [float(v) for v in tilt_alphas_rad],
        "tilt_s_px_cam1": [float(v) for v in tilt_s_px_cam1],
        "tilt_s_px_cam2": [float(v) for v in tilt_s_px_cam2],
        "metadata": dict(metadata or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
