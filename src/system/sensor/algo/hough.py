from dataclasses import dataclass, field
import math

import numpy as np

from src.shared import CameraObservation, CameraParams, CAMERA_PRESETS_REGISTRY
from .base import DVSLineAlgorithm

try:
    import numba as _numba  # type: ignore
    njit = _numba.njit
except ModuleNotFoundError:
    def njit(*_args, **_kwargs):  # type: ignore
        def _decorator(fn):
            return fn
        return _decorator


# ── Internal Hough state ──────────────────────────────────────────────────────

@dataclass
class HoughTrackerParams:
    mixing_factor:    float = 0.02
    inlier_stddev_px: float = 4.0
    min_determinant:  float = 1e-6


@dataclass
class HoughQuadraticState:
    quadratic_m2: float = 0.0
    cross_mb:     float = 0.0
    quadratic_b2: float = 0.0
    linear_m:     float = 0.0
    linear_b:     float = 0.0


# ── Registry Params ───────────────────────────────────────────────────────────

@dataclass
class HoughLineParams:
    cam_params:       CameraParams
    mixing_factor:    float
    inlier_stddev_px: float
    min_determinant:  float
    max_events:       int | None = None
    quadratic_m2:     float = 0.0
    cross_mb:         float = 0.0
    quadratic_b2:     float = 0.0
    linear_m:         float = 0.0
    linear_b:         float = 0.0


HOUGH_PRESETS = {
    "default": {
        "cam_params":       "default:default",
        "mixing_factor":    0.02,
        "inlier_stddev_px": 4.0,
        "min_determinant":  1e-6,
        "quadratic_m2":     0.0,
        "cross_mb":         0.0,
        "quadratic_b2":     0.0,
        "linear_m":         0.0,
        "linear_b":         0.0,
    }
}


@njit(cache=True)
def _hough_update_events_jit(
    xs_centered, ys_centered,
    q_m2, cross_mb, q_b2, lin_m, lin_b,
    mixing_factor, inv_2sigma2, min_determinant,
):
    """Numba-compiled inner loop for the recursive Hough tracker."""
    for i in range(len(xs_centered)):
        det = 4.0 * q_m2 * q_b2 - cross_mb * cross_mb
        if abs(det) < min_determinant:
            continue
        intercept = (lin_m * cross_mb - 2.0 * q_m2 * lin_b) / det
        slope = (cross_mb * lin_b - 2.0 * q_b2 * lin_m) / det

        predicted_x = intercept + ys_centered[i] * slope
        residual = xs_centered[i] - predicted_x
        weight = math.exp(-residual * residual * inv_2sigma2)

        dec = 1.0 - mixing_factor * weight
        q_m2     *= dec
        cross_mb *= dec
        q_b2     *= dec
        lin_m    *= dec
        lin_b    *= dec

        yi = ys_centered[i]
        xi = xs_centered[i]
        q_m2     += weight * (yi * yi)
        cross_mb += weight * (2.0 * yi)
        q_b2     += weight
        lin_m    += weight * (-2.0 * xi * yi)
        lin_b    += weight * (-2.0 * xi)

    return q_m2, cross_mb, q_b2, lin_m, lin_b


class PaperHoughLineAlgorithm(DVSLineAlgorithm):
    """Recursive Hough line tracker. Accepts ``HoughLineParams`` from the registry."""

    def __init__(self, params: HoughLineParams):
        cam = params.cam_params
        self.width  = int(cam.DAVIS346_WIDTH)
        self.height = int(cam.DAVIS346_HEIGHT)
        self.cx = self.width  / 2
        self.cy = self.height / 2

        self.params = HoughTrackerParams(
            mixing_factor    = params.mixing_factor,
            inlier_stddev_px = params.inlier_stddev_px,
            min_determinant  = params.min_determinant,
        )
        self.max_events = params.max_events

        sigma = self.params.inlier_stddev_px
        self._inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)

        self.state = HoughQuadraticState()
        self.current_centered_line: CameraObservation | None = None
        self.reset()

    def _solve_centered_line(self) -> CameraObservation | None:
        determinant = (
            4.0 * self.state.quadratic_m2 * self.state.quadratic_b2
            - self.state.cross_mb * self.state.cross_mb
        )
        if abs(determinant) < self.params.min_determinant:
            return None

        centered_intercept = (
            self.state.linear_m * self.state.cross_mb
            - 2.0 * self.state.quadratic_m2 * self.state.linear_b
        ) / determinant
        slope = (
            self.state.cross_mb * self.state.linear_b
            - 2.0 * self.state.quadratic_b2 * self.state.linear_m
        ) / determinant

        return CameraObservation(slope=slope, intercept=centered_intercept)

    def _accumulate_weighted_event(self, x_centered, y_centered, weight):
        self.state.quadratic_m2 += weight * (y_centered * y_centered)
        self.state.cross_mb     += weight * (2.0 * y_centered)
        self.state.quadratic_b2 += weight
        self.state.linear_m     += weight * (-2.0 * x_centered * y_centered)
        self.state.linear_b     += weight * (-2.0 * x_centered)

    def _seed_vertical_line(self):
        self.state = HoughQuadraticState()
        bootstrap_x = 0.0
        for y_centered in (-self.cy, self.height - 1 - self.cy):
            self._accumulate_weighted_event(bootstrap_x, y_centered, weight=1.0)
        self.current_centered_line = self._solve_centered_line()

    def _current_pixel_observation(self) -> CameraObservation | tuple[None, None]:
        centered_line = self._solve_centered_line()
        self.current_centered_line = centered_line
        if centered_line is None:
            return None, None
        pixel_intercept = (
            centered_line.intercept + self.cx - centered_line.slope * self.cy
        )
        return CameraObservation(slope=centered_line.slope, intercept=pixel_intercept)

    def update(self, events_np):
        if events_np is None or len(events_np) == 0:
            return self._current_pixel_observation()

        if self.max_events is not None and len(events_np) > self.max_events:
            events_np = events_np[-self.max_events:]

        xs_centered = events_np["x"].astype(np.float64) - self.cx
        ys_centered = events_np["y"].astype(np.float64) - self.cy

        s = self.state
        (s.quadratic_m2, s.cross_mb, s.quadratic_b2,
         s.linear_m, s.linear_b) = _hough_update_events_jit(
            xs_centered, ys_centered,
            s.quadratic_m2, s.cross_mb, s.quadratic_b2,
            s.linear_m, s.linear_b,
            self.params.mixing_factor, self._inv_2sigma2,
            self.params.min_determinant,
        )
        return self._current_pixel_observation()

    def reset(self):
        self._seed_vertical_line()
