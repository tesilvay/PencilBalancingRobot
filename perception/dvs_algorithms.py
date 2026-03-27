import math

import numpy as np

from perception.dvs_camera_reader import DAVIS346_WIDTH, DAVIS346_HEIGHT

from core.sim_types import CameraObservation, HoughQuadraticState, HoughTrackerParams


try:
    import numba as _numba  # type: ignore

    njit = _numba.njit
except ModuleNotFoundError:  # pragma: no cover
    # Allow running visualization/tools without numba installed.
    def njit(*_args, **_kwargs):  # type: ignore
        def _decorator(fn):
            return fn

        return _decorator


def mask_events_below_line(events_np: np.ndarray, mask_line_y: int, frame_height: int) -> np.ndarray:
    """
    Keep only events with y < mask_line_y.

    Edge cases:
    - mask_line_y <= 0: return empty array (same dtype)
    - mask_line_y >= frame_height: no masking (return input)
    """
    if events_np is None:
        return events_np
    if mask_line_y >= frame_height:
        return events_np
    if mask_line_y <= 0:
        return events_np[:0]
    return events_np[events_np["y"] < mask_line_y]


def line_x_at_pixel_y(obs_px: CameraObservation, y: float) -> float:
    """Evaluate pixel-space line model x = slope*y + intercept at a given y."""
    return float(obs_px.slope) * float(y) + float(obs_px.intercept)

@njit(cache=True)
def _hough_update_events_jit(
    xs_centered, ys_centered,
    q_m2, cross_mb, q_b2, lin_m, lin_b,
    mixing_factor, inv_2sigma2, min_determinant,
):
    """Numba-compiled inner loop for the recursive Hough tracker.

    Mirrors Conradt/PencilBalancer.java polyAddEventX/Y: per-event solve,
    Gaussian inlier weight, adaptive forgetting, and accumulation.
    """
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


class DVSLineAlgorithm:
    """Base class for DVS line estimation algorithms."""

    def update(self, events_np):
        raise NotImplementedError

    def reset(self):
        pass


class SurfaceRegressionAlgorithm(DVSLineAlgorithm):
    """
    Activity surface + linear regression.
    """

    def __init__(self, width, height, decay=0.8, threshold=0.5, min_points=50):

        self.W = width
        self.H = height

        self.decay = decay
        self.threshold = threshold
        self.min_points = min_points

        self.surface = np.zeros((height, width), dtype=np.float32)

    def update(self, events_np):

        xs = events_np['x']
        ys = events_np['y']

        # decay previous activity
        self.surface *= self.decay

        # accumulate events
        self.surface[ys, xs] += 1.0

        mask = self.surface > self.threshold
        points = np.column_stack(np.where(mask))  # (y,x)

        if len(points) < self.min_points:
            return None, None

        ys_pts = points[:, 0].astype(np.float32)
        xs_pts = points[:, 1].astype(np.float32)

        N = len(xs_pts)

        S_y = np.sum(ys_pts)
        S_yy = np.sum(ys_pts * ys_pts)
        S_x = np.sum(xs_pts)
        S_xy = np.sum(xs_pts * ys_pts)

        denom = N * S_yy - S_y * S_y

        if abs(denom) < 1e-6:
            return None, None

        s = (N * S_xy - S_y * S_x) / denom
        b = (S_x - s * S_y) / N

        return s, b


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

        # Match the Java bootstrap: seed a centered vertical line immediately
        # so the first real events have a valid line to compare against.
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


class SamLineAlgorithm(DVSLineAlgorithm):
    """
    OLS line fit on event coordinates (from sam_cam.py).
    Line: x = slope * y + intercept.
    Fits directly to event batch; no surface accumulation.
    """

    def __init__(self, width=DAVIS346_WIDTH, height=DAVIS346_HEIGHT, min_points=50):
        self.W = width
        self.H = height
        self.min_points = min_points

    def update(self, events_np):
        xs = events_np["x"].astype(np.float32)
        ys = events_np["y"].astype(np.float32)

        if len(xs) < self.min_points:
            return None, None

        N = len(xs)
        S_y = np.sum(ys)
        S_yy = np.sum(ys * ys)
        S_x = np.sum(xs)
        S_xy = np.sum(xs * ys)

        denom = N * S_yy - S_y * S_y

        if abs(denom) < 1e-6:
            return None, None

        slope = (N * S_xy - S_y * S_x) / denom
        intercept = (S_x - slope * S_y) / N

        return CameraObservation(slope=slope, intercept=intercept)

