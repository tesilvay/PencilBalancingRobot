from dataclasses import dataclass
import math

import numpy as np

from perception.dvs_camera_reader import DAVIS346_WIDTH, DAVIS346_HEIGHT
from core.sim_types import CameraObservation, HoughQuadraticState, HoughTrackerParams

from .base import DVSLineAlgorithm

try:
    import numba as _numba  # type: ignore

    njit = _numba.njit
except ModuleNotFoundError:  # pragma: no cover
    def njit(*_args, **_kwargs):  # type: ignore
        def _decorator(fn):
            return fn

        return _decorator


@dataclass
class HoughLineParams:
    mixing_factor:    float
    inlier_stddev_px: float
    min_determinant:  float
    max_events:       int | None = None


HOUGH_PRESETS = {
    "default": {
        "mixing_factor":    0.02,
        "inlier_stddev_px": 4.0,
        "min_determinant":  1e-6,
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
        q_m2 *= dec
        cross_mb *= dec
        q_b2 *= dec
        lin_m *= dec
        lin_b *= dec

        yi = ys_centered[i]
        xi = xs_centered[i]
        q_m2 += weight * (yi * yi)
        cross_mb += weight * (2.0 * yi)
        q_b2 += weight
        lin_m += weight * (-2.0 * xi * yi)
        lin_b += weight * (-2.0 * xi)

    return q_m2, cross_mb, q_b2, lin_m, lin_b


class PaperHoughLineAlgorithm(DVSLineAlgorithm):
    """
    Readable Python port of the original Java recursive Hough tracker.

    The original implementation maintained a quadratic objective over line
    parameters and updated it per event using:
    - a Gaussian inlier weight based on distance to the current line estimate
    - an adaptive forgetting factor tied to that inlier weight

    This implementation preserves that behavior while separating the update
    into small, named steps.
    """

    def __init__(self, width=346, height=260, params: HoughTrackerParams | None = None,
                 max_events: int | None = None):
        self.width = width
        self.height = height
        self.cx = width / 2
        self.cy = height / 2
        self.params = HoughTrackerParams() if params is None else params
        self.max_events = max_events

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

    def _event_residual(self, x_centered: float, y_centered: float, estimate: CameraObservation) -> float:
        predicted_x = estimate.intercept + y_centered * estimate.slope
        return x_centered - predicted_x

    def _gaussian_inlier_weight(self, residual: float) -> float:
        sigma = self.params.inlier_stddev_px
        return float(np.exp(-(residual * residual) / (2.0 * sigma * sigma)))

    def _adaptive_decay(self, weight: float) -> float:
        return 1.0 - self.params.mixing_factor * weight

    def _apply_forgetting(self, decay_factor: float) -> None:
        self.state.quadratic_m2 *= decay_factor
        self.state.cross_mb *= decay_factor
        self.state.quadratic_b2 *= decay_factor
        self.state.linear_m *= decay_factor
        self.state.linear_b *= decay_factor

    def _accumulate_weighted_event(self, x_centered: float, y_centered: float, weight: float) -> None:
        self.state.quadratic_m2 += weight * (y_centered * y_centered)
        self.state.cross_mb += weight * (2.0 * y_centered)
        self.state.quadratic_b2 += weight
        self.state.linear_m += weight * (-2.0 * x_centered * y_centered)
        self.state.linear_b += weight * (-2.0 * x_centered)

    def _seed_vertical_line(self) -> None:
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

        pixel_intercept = centered_line.intercept + self.cx - centered_line.slope * self.cy
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
