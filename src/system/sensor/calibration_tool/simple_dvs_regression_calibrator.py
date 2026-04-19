from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from src.shared import WorkspaceParams, CameraObservation, CameraPair, default_camera_params
from src.system.sensor.observation_model.camera_model import CameraModel
from src.system.sensor.algo.dvs_algorithms import mask_events_below_line
from src.system.sensor.reader.dvs_camera_reader import (
    DAVIS346_HEIGHT,
    DAVIS346_WIDTH,
    DVSReader,
    discover_devices,
)
from src.system.sensor.observation_model.simple_dvs_regression_model import (
    default_affine_calibration_path,
    save_affine_v1_calibration,
    save_bilinear_v2_calibration,
    save_tps_v3_calibration,
    SimpleDVSRegressionModel,
)
from src.experiment.realtime_visualizer import OneDvsVisualizer, OneDvsVisualizerParams


_MIDPOINT_PX = DAVIS346_WIDTH / 2


def _predict_pixel_intercepts(
    model: SimpleDVSRegressionModel | None,
    px_m: float,
    py_m: float,
) -> tuple[float, float]:
    """
    Predict (b1_px, b2_px) for a target world position using the current calibration.

    v3 TPS: evaluate forward RBF directly — O(N) but exact.
    v2 bilinear: solve the 2x2 system.
    v1 affine: direct per-axis inversion.

    Returns (b1_px, b2_px) clamped to [0, DAVIS346_WIDTH-1].
    Falls back to midpoint if model is None or singular.
    """
    fallback = _MIDPOINT_PX

    if model is None:
        return fallback, fallback

    try:
        if model.pos_v3 is not None:
            pt = np.array([[px_m, py_m]])
            b1 = float(model._rbf_b1(pt)[0])  # type: ignore[attr-defined]
            b2 = float(model._rbf_b2(pt)[0])  # type: ignore[attr-defined]
        elif model.pos_v2 is not None:
            v2 = model.pos_v2
            A = np.array([[v2.a1, v2.a2], [v2.c2, v2.c1]], dtype=float)
            rhs = np.array([px_m - v2.a0, py_m - v2.c0], dtype=float)
            if abs(float(np.linalg.det(A))) < 1e-12:
                return fallback, fallback
            b1, b2 = np.linalg.solve(A, rhs)
        elif model.cam1 is not None and model.cam2 is not None:
            k1, off1 = model.cam1.k_pos, model.cam1.b_pos
            k2, off2 = model.cam2.k_pos, model.cam2.b_pos
            b1 = (px_m - off1) / k1 if abs(k1) > 1e-12 else fallback
            b2 = (py_m - off2) / k2 if abs(k2) > 1e-12 else fallback
        else:
            return fallback, fallback
    except Exception:
        return fallback, fallback

    lo, hi = 0.0, float(DAVIS346_WIDTH - 1)
    return float(np.clip(b1, lo, hi)), float(np.clip(b2, lo, hi))


def _predict_tilt_slopes(
    model: SimpleDVSRegressionModel | None,
    alpha_rad: float,
    px_m: float = 0.0,
    py_m: float = 0.0,
) -> tuple[float, float]:
    """
    Predict (s1_px, s2_px) for a given tilt angle and world position.

    v3 TPS: queries the 3D tilt RBF at (px_m, py_m, alpha_rad) — position-aware.
    v1/v2: inverts the 1D piecewise-linear tilt table — position-blind.
    Falls back to 0.0 if no model or table has fewer than 2 points.
    """
    if model is None:
        return 0.0, 0.0

    if model.pos_v3 is not None:
        try:
            pt = np.array([[px_m, py_m, alpha_rad]])
            s1 = float(model._rbf_s1(pt)[0])  # type: ignore[attr-defined]
            s2 = float(model._rbf_s2(pt)[0])  # type: ignore[attr-defined]
            return s1, s2
        except Exception:
            return 0.0, 0.0

    if model.cam1 is None or model.cam2 is None:
        return 0.0, 0.0

    def invert(slopes: np.ndarray, alphas: np.ndarray, target: float) -> float:
        if len(slopes) < 2:
            return 0.0
        order = np.argsort(alphas, kind="mergesort")
        return float(np.interp(target, alphas[order], slopes[order]))

    slopes1, alphas1 = model.cam1.tilt_interp
    slopes2, alphas2 = model.cam2.tilt_interp
    return invert(slopes1, alphas1, alpha_rad), invert(slopes2, alphas2, alpha_rad)


def x_positions_from_safe_radius(safe_radius_m: float, step_m: float = 0.01) -> list[float]:
    r = float(safe_radius_m)
    step = float(step_m)
    if step <= 0 or not math.isfinite(step):
        raise ValueError("step_m must be positive and finite")
    n = int(math.floor((r - 1e-9) / step))
    if n < 0:
        n = 0
    return [k * step for k in range(-n, n + 1)]


def xy_grid_positions(safe_radius_m: float, step_m: float) -> list[tuple[float, float]]:
    """All (px_m, py_m) grid points within the circular workspace, sorted by (py, px)."""
    r = float(safe_radius_m)
    step = float(step_m)
    n = int(math.floor((r - 1e-9) / step))
    positions = []
    for j in range(-n, n + 1):
        for i in range(-n, n + 1):
            x, y = i * step, j * step
            if math.hypot(x, y) <= r + 1e-9:
                positions.append((x, y))
    return positions


# Default tilt grid for both cameras (deg), converted to rad at runtime.
DEFAULT_TILT_CALIB_DEGS: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0)

DEFAULT_WORKSPACE = WorkspaceParams(x_ref=0.0, y_ref=0.0, safe_radius=0.068)
DEFAULT_CAMERA_PARAMS = default_camera_params()


def tilt_degs_to_rads(degs: list[float] | tuple[float, ...]) -> list[float]:
    return [float(np.deg2rad(d)) for d in degs]


def manual_state_to_cam1_pair(state: ManualLineState, mask_y: int, cam: CameraModel) -> CameraPair:
    obs_px = state.to_obs_px(mask_y=mask_y)
    n = cam.pixel_to_camnorm(obs_px)
    return CameraPair(cam1=n, cam2=CameraObservation(slope=0.0, intercept=0.0))


def manual_state_to_cam2_pair(state: ManualLineState, mask_y: int, cam: CameraModel) -> CameraPair:
    obs_px = state.to_obs_px(mask_y=mask_y)
    n = cam.pixel_to_camnorm(obs_px)
    return CameraPair(cam1=CameraObservation(slope=0.0, intercept=0.0), cam2=n)


# =========================
# DATA LAYER (pure storage)
# =========================

@dataclass
class ManualLineState:
    slope_px: float = 0.0
    x_at_mask_px: float = 0.0

    def to_obs_px(self, mask_y: int) -> tuple[float, float]:
        b_px = float(self.x_at_mask_px) - float(self.slope_px) * float(mask_y)
        return CameraObservation(slope=float(self.slope_px), intercept=float(b_px))

@dataclass
class X_Samples:
    x_pos_m: list[float]
    x_at_mask_px: list[float | None]
    mask_y: int


@dataclass
class Y_Samples:
    y_pos_m: list[float]
    x_at_mask_px: list[float | None]
    mask_y: int


@dataclass
class AX_Samples:
    ax_pos_rad: list[float]
    s_px: list[float | None]


@dataclass
class AY_Samples:
    ay_pos_rad: list[float]
    s_px: list[float | None]


@dataclass
class XY_Grid_Samples:
    """2D grid data: both cameras' intercepts at each (px_m, py_m) position."""
    positions: list[tuple[float, float]]
    b1_px: list[float | None]   # cam1 x_at_mask for each position
    b2_px: list[float | None]   # cam2 x_at_mask for each position
    mask_y_cam1: int
    mask_y_cam2: int


@dataclass
class TiltGrid_Samples:
    """
    3D tilt samples: at each (px_m, py_m, alpha_rad), record the observed slope_px.

    Flat lists length = n_positions * n_angles, position-major / angle-inner:
      flat_idx = pi * len(alphas_rad) + ai

    Cam1 owns alpha_x (rotation about x-axis → visible as slope on cam1).
    Cam2 owns alpha_y (rotation about y-axis → visible as slope on cam2).
    """
    positions: list[tuple[float, float]]  # length n_positions
    alphas_rad: list[float]               # length n_angles
    s_px_cam1: list[float | None]         # length n_positions * n_angles
    s_px_cam2: list[float | None]         # length n_positions * n_angles
    mask_y_cam1: int
    mask_y_cam2: int


@dataclass
class Samples:
    x: X_Samples
    y: Y_Samples
    ax: AX_Samples
    ay: AY_Samples


def build_samples(args, init_state, position_targets, tilt_degs, existing_model=None):
    tilt_rads = tilt_degs_to_rads(tilt_degs)
    n_tilt = len(tilt_rads)

    # Pre-fill pixel intercept predictions from the existing model for each position.
    # v1 X stage sweeps along x with y=0; Y stage sweeps along y with x=0.
    x_at_mask_cam1 = [
        _predict_pixel_intercepts(existing_model, pos, 0.0)[0]
        for pos in position_targets
    ]
    x_at_mask_cam2 = [
        _predict_pixel_intercepts(existing_model, 0.0, pos)[1]
        for pos in position_targets
    ]

    return Samples(
        x=X_Samples(
            x_pos_m=list(position_targets),
            x_at_mask_px=x_at_mask_cam1,
            mask_y=args.mask_y_cam1,
        ),
        y=Y_Samples(
            y_pos_m=list(position_targets),
            x_at_mask_px=x_at_mask_cam2,
            mask_y=args.mask_y_cam2,
        ),
        ax=AX_Samples(
            ax_pos_rad=list(tilt_rads),
            s_px=[init_state.slope_px] + [None] * (n_tilt - 1),
        ),
        ay=AY_Samples(
            ay_pos_rad=list(tilt_rads),
            s_px=[init_state.slope_px] + [None] * (n_tilt - 1),
        ),
    )


def _affine_sample_arrays(samples: Samples) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float]]:
    """Raises ValueError if any sample slot is unset (None) or lengths disagree."""

    def floats_no_none(vals: list[float | None], label: str) -> list[float]:
        if any(v is None for v in vals):
            raise ValueError(f"Incomplete calibration: unset entries in {label}")
        return [float(v) for v in vals]  # type: ignore[misc]

    x, y, ax, ay = samples.x, samples.y, samples.ax, samples.ay
    if len(x.x_pos_m) != len(x.x_at_mask_px):
        raise ValueError("X position samples: x_pos_m and x_at_mask_px length mismatch")
    if len(y.y_pos_m) != len(y.x_at_mask_px):
        raise ValueError("Y position samples: y_pos_m and x_at_mask_px length mismatch")
    if len(ax.ax_pos_rad) != len(ax.s_px):
        raise ValueError("AX samples: length mismatch")
    if len(ay.ay_pos_rad) != len(ay.s_px):
        raise ValueError("AY samples: length mismatch")

    x_at_1 = floats_no_none(x.x_at_mask_px, "cam1 x_at_mask_px")
    x_at_2 = floats_no_none(y.x_at_mask_px, "cam2 x_at_mask_px")
    s_ax = floats_no_none(ax.s_px, "cam1 tilt slope_px")
    s_ay = floats_no_none(ay.s_px, "cam2 tilt slope_px")

    return (
        x_at_1,
        [float(v) for v in x.x_pos_m],
        x_at_2,
        [float(v) for v in y.y_pos_m],
        s_ax,
        [float(v) for v in ax.ax_pos_rad],
        s_ay,
        [float(v) for v in ay.ay_pos_rad],
    )


def _v2_sample_arrays(
    grid: XY_Grid_Samples,
    ax_samples: AX_Samples,
    ay_samples: AY_Samples,
) -> tuple[
    list[tuple[float, float]], list[float], list[float],
    list[float], list[float], list[float], list[float],
]:
    """Raises ValueError if any grid slot is unset."""
    def floats_no_none(vals: list[float | None], label: str) -> list[float]:
        if any(v is None for v in vals):
            raise ValueError(f"Incomplete calibration: unset entries in {label}")
        return [float(v) for v in vals]  # type: ignore[misc]

    b1 = floats_no_none(grid.b1_px, "grid cam1 b1_px")
    b2 = floats_no_none(grid.b2_px, "grid cam2 b2_px")
    s_ax = floats_no_none(ax_samples.s_px, "cam1 tilt slope_px")
    s_ay = floats_no_none(ay_samples.s_px, "cam2 tilt slope_px")
    return (
        list(grid.positions),
        b1,
        b2,
        s_ax,
        [float(v) for v in ax_samples.ax_pos_rad],
        s_ay,
        [float(v) for v in ay_samples.ay_pos_rad],
    )


def _v3_sample_arrays(
    grid: XY_Grid_Samples,
    tilt: TiltGrid_Samples,
) -> tuple[
    list[tuple[float, float]], list[float], list[float],
    list[tuple[float, float]], list[float], list[float], list[float],
]:
    """Raises ValueError if any grid or tilt slot is unset (None)."""
    def floats_no_none(vals: list[float | None], label: str) -> list[float]:
        if any(v is None for v in vals):
            raise ValueError(f"Incomplete calibration: unset entries in {label}")
        return [float(v) for v in vals]  # type: ignore[misc]

    b1 = floats_no_none(grid.b1_px, "grid cam1 b1_px")
    b2 = floats_no_none(grid.b2_px, "grid cam2 b2_px")
    s1 = floats_no_none(tilt.s_px_cam1, "tilt cam1 s_px")
    s2 = floats_no_none(tilt.s_px_cam2, "tilt cam2 s_px")
    return (
        list(grid.positions), b1, b2,
        list(tilt.positions), list(tilt.alphas_rad), s1, s2,
    )


def _calibration_title(prefix: str, idx: int, total: int, extras: str, adjust_hint: str) -> str:
    return (
        f"{prefix} | {idx + 1}/{total} | {extras} | {adjust_hint} | "
        "SPACE save | Z back | Q quit"
    )


# =========================
# STAGE LAYER (semantics)
# =========================


class Stage:
    """Owns mask, camera model, and how state maps to a CameraPair for the renderer."""

    def __init__(
        self,
        name: str,
        cam: str,
        *,
        cam_index: int,
        surface_index: int,
        cam_model: CameraModel,
    ) -> None:
        self.name = name
        self.cam = cam
        self.cam_index = int(cam_index)
        self.surface_index = int(surface_index)
        self.cam_model = cam_model

    def measure(self, state: ManualLineState) -> Any:
        raise NotImplementedError

    def title(self, idx: int, total: int, state: ManualLineState) -> str:
        raise NotImplementedError

    def mask(self) -> int:
        raise NotImplementedError

    def apply_key(self, state: ManualLineState, key: str, step_x: float, step_s: float) -> None:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def load_sample_into_state(self, state: ManualLineState, idx: int) -> None:
        raise NotImplementedError

    def save_state_to_sample(self, idx: int, state: ManualLineState) -> None:
        raise NotImplementedError


class XStage(Stage):
    def __init__(self, name: str, cam: str, cam_index: int, surface_index: int, cam_model: CameraModel, samples: X_Samples) -> None:
        super().__init__(name, cam, cam_index=cam_index, surface_index=surface_index, cam_model=cam_model)
        self.samples = samples

    def measure(self, state: ManualLineState):
        return manual_state_to_cam1_pair(state, self.samples.mask_y, self.cam_model)

    def mask(self) -> int:
        return int(self.samples.mask_y)

    def title(self, idx: int, total: int, state: ManualLineState) -> str:
        obs = state.to_obs_px(mask_y=self.samples.mask_y)
        x = self.samples.x_pos_m[idx]
        extras = (
            f"target X={x * 100:+.1f} cm | "
            f"b1_px={obs.intercept:+.2f} | x@mask={state.x_at_mask_px:+.1f}"
        )
        return _calibration_title("X cam1", idx, total, extras, "A/D move")

    def apply_key(self, state: ManualLineState, key: str, step_x: float, step_s: float) -> None:
        del step_s
        if key == "a":
            state.x_at_mask_px -= step_x
        elif key == "d":
            state.x_at_mask_px += step_x

    def size(self) -> int:
        return len(self.samples.x_pos_m)

    def load_sample_into_state(self, state: ManualLineState, idx: int) -> None:
        state.slope_px = 0.0  # position calibration always uses a vertical line
        v = self.samples.x_at_mask_px[idx]
        if v is not None:
            state.x_at_mask_px = float(v)

    def save_state_to_sample(self, idx: int, state: ManualLineState) -> None:
        self.samples.x_at_mask_px[idx] = state.x_at_mask_px


class YStage(Stage):
    def __init__(self, name: str, cam: str, cam_index: int, surface_index: int, cam_model: CameraModel, samples: Y_Samples) -> None:
        super().__init__(name, cam, cam_index=cam_index, surface_index=surface_index, cam_model=cam_model)
        self.samples = samples

    def measure(self, state: ManualLineState):
        return manual_state_to_cam2_pair(state, self.samples.mask_y, self.cam_model)

    def mask(self) -> int:
        return int(self.samples.mask_y)

    def title(self, idx: int, total: int, state: ManualLineState) -> str:
        obs = state.to_obs_px(mask_y=self.samples.mask_y)
        y = self.samples.y_pos_m[idx]
        extras = (
            f"target Y={y * 100:+.1f} cm | "
            f"b2_px={obs.intercept:+.2f} | x@mask={state.x_at_mask_px:+.1f}"
        )
        return _calibration_title("Y cam2", idx, total, extras, "A/D move")

    def apply_key(self, state: ManualLineState, key: str, step_x: float, step_s: float) -> None:
        del step_s
        if key == "a":
            state.x_at_mask_px -= step_x
        elif key == "d":
            state.x_at_mask_px += step_x

    def size(self) -> int:
        return len(self.samples.y_pos_m)

    def load_sample_into_state(self, state: ManualLineState, idx: int) -> None:
        state.slope_px = 0.0  # position calibration always uses a vertical line
        v = self.samples.x_at_mask_px[idx]
        if v is not None:
            state.x_at_mask_px = float(v)

    def save_state_to_sample(self, idx: int, state: ManualLineState) -> None:
        self.samples.x_at_mask_px[idx] = state.x_at_mask_px


class AXStage(Stage):
    """AX on camera 1. Intercept locked to image midpoint; A/D adjust slope only."""

    def __init__(self, cam_model: CameraModel, samples: AX_Samples, mask_y_cam1: int) -> None:
        super().__init__("AX", "x", cam_index=0, surface_index=0, cam_model=cam_model)
        self.samples = samples
        self._mask_y_val = int(mask_y_cam1)

    def measure(self, state: ManualLineState):
        return manual_state_to_cam1_pair(state, self._mask_y_val, self.cam_model)

    def mask(self) -> int:
        return self._mask_y_val

    def title(self, idx: int, total: int, state: ManualLineState) -> str:
        tgt = self.samples.ax_pos_rad[idx]
        deg = math.degrees(tgt)
        extras = f"set alpha_x={deg:+.1f} | s_px={state.slope_px:+.4f}"
        return _calibration_title("AX cam1", idx, total, extras, "A/D slope")

    def apply_key(self, state: ManualLineState, key: str, step_x: float, step_s: float) -> None:
        del step_x
        if key == "a":
            state.slope_px += step_s
        elif key == "d":
            state.slope_px -= step_s

    def size(self) -> int:
        return len(self.samples.ax_pos_rad)

    def load_sample_into_state(self, state: ManualLineState, idx: int) -> None:
        state.x_at_mask_px = _MIDPOINT_PX  # tilt calibration always uses midpoint intercept
        v = self.samples.s_px[idx]
        if v is not None:
            state.slope_px = float(v)

    def save_state_to_sample(self, idx: int, state: ManualLineState) -> None:
        self.samples.s_px[idx] = state.slope_px


class AYStage(Stage):
    """AY on camera 2. Intercept locked to image midpoint; A/D adjust slope only."""

    def __init__(self, cam_model: CameraModel, samples: AY_Samples, mask_y_cam2: int) -> None:
        super().__init__("AY", "y", cam_index=1, surface_index=1, cam_model=cam_model)
        self.samples = samples
        self._mask_y_val = int(mask_y_cam2)

    def measure(self, state: ManualLineState):
        return manual_state_to_cam2_pair(state, self._mask_y_val, self.cam_model)

    def mask(self) -> int:
        return self._mask_y_val

    def title(self, idx: int, total: int, state: ManualLineState) -> str:
        tgt = self.samples.ay_pos_rad[idx]
        deg = math.degrees(tgt)
        extras = f"set alpha_y={deg:+.1f} | s_px={state.slope_px:+.4f}"
        return _calibration_title("AY cam2", idx, total, extras, "A/D slope")

    def apply_key(self, state: ManualLineState, key: str, step_x: float, step_s: float) -> None:
        del step_x
        if key == "a":
            state.slope_px += step_s
        elif key == "d":
            state.slope_px -= step_s

    def size(self) -> int:
        return len(self.samples.ay_pos_rad)

    def load_sample_into_state(self, state: ManualLineState, idx: int) -> None:
        state.x_at_mask_px = _MIDPOINT_PX  # tilt calibration always uses midpoint intercept
        v = self.samples.s_px[idx]
        if v is not None:
            state.slope_px = float(v)

    def save_state_to_sample(self, idx: int, state: ManualLineState) -> None:
        self.samples.s_px[idx] = state.slope_px


class XYGridCam1Stage(Stage):
    """
    Records cam1 intercept (b1) at each (px_m, py_m) grid position.
    Slope is always 0 — position calibration uses vertical lines only.
    """

    def __init__(self, cam_model: CameraModel, grid: XY_Grid_Samples) -> None:
        super().__init__("XYGrid-cam1", "x", cam_index=0, surface_index=0, cam_model=cam_model)
        self.grid = grid

    def measure(self, state: ManualLineState):
        return manual_state_to_cam1_pair(state, self.grid.mask_y_cam1, self.cam_model)

    def mask(self) -> int:
        return int(self.grid.mask_y_cam1)

    def title(self, idx: int, total: int, state: ManualLineState) -> str:
        px_m, py_m = self.grid.positions[idx]
        obs = state.to_obs_px(mask_y=self.grid.mask_y_cam1)
        extras = (
            f"Move to X={px_m * 100:+.1f}cm Y={py_m * 100:+.1f}cm | cam1 | "
            f"b1_px={obs.intercept:+.2f} | x@mask={state.x_at_mask_px:+.1f}"
        )
        return _calibration_title("XYGrid", idx, total, extras, "A/D move")

    def apply_key(self, state: ManualLineState, key: str, step_x: float, step_s: float) -> None:
        del step_s
        if key == "a":
            state.x_at_mask_px -= step_x
        elif key == "d":
            state.x_at_mask_px += step_x

    def size(self) -> int:
        return len(self.grid.positions)

    def load_sample_into_state(self, state: ManualLineState, idx: int) -> None:
        state.slope_px = 0.0  # always vertical
        v = self.grid.b1_px[idx]
        if v is not None:
            state.x_at_mask_px = float(v)

    def save_state_to_sample(self, idx: int, state: ManualLineState) -> None:
        self.grid.b1_px[idx] = state.x_at_mask_px


class XYGridCam2Stage(Stage):
    """
    Records cam2 intercept (b2) at each (px_m, py_m) grid position.
    Slope is always 0 — position calibration uses vertical lines only.
    """

    def __init__(self, cam_model: CameraModel, grid: XY_Grid_Samples) -> None:
        super().__init__("XYGrid-cam2", "y", cam_index=1, surface_index=1, cam_model=cam_model)
        self.grid = grid

    def measure(self, state: ManualLineState):
        return manual_state_to_cam2_pair(state, self.grid.mask_y_cam2, self.cam_model)

    def mask(self) -> int:
        return int(self.grid.mask_y_cam2)

    def title(self, idx: int, total: int, state: ManualLineState) -> str:
        px_m, py_m = self.grid.positions[idx]
        obs = state.to_obs_px(mask_y=self.grid.mask_y_cam2)
        extras = (
            f"Move to X={px_m * 100:+.1f}cm Y={py_m * 100:+.1f}cm | cam2 | "
            f"b2_px={obs.intercept:+.2f} | x@mask={state.x_at_mask_px:+.1f}"
        )
        return _calibration_title("XYGrid", idx, total, extras, "A/D move")

    def apply_key(self, state: ManualLineState, key: str, step_x: float, step_s: float) -> None:
        del step_s
        if key == "a":
            state.x_at_mask_px -= step_x
        elif key == "d":
            state.x_at_mask_px += step_x

    def size(self) -> int:
        return len(self.grid.positions)

    def load_sample_into_state(self, state: ManualLineState, idx: int) -> None:
        state.slope_px = 0.0  # always vertical
        v = self.grid.b2_px[idx]
        if v is not None:
            state.x_at_mask_px = float(v)

    def save_state_to_sample(self, idx: int, state: ManualLineState) -> None:
        self.grid.b2_px[idx] = state.x_at_mask_px


class TiltGridCam1Stage(Stage):
    """
    Records cam1 slope at each (position, alpha_x) grid point.
    Positions iterate outer, angles iterate inner — move robot once per 5 angles.
    Intercept locked to midpoint; A/D adjust slope only.
    """

    def __init__(self, cam_model: CameraModel, tilt: TiltGrid_Samples) -> None:
        super().__init__("TiltGrid-cam1", "x", cam_index=0, surface_index=0, cam_model=cam_model)
        self.tilt = tilt

    def _unpack(self, flat_idx: int) -> tuple[int, int]:
        n_a = len(self.tilt.alphas_rad)
        return flat_idx // n_a, flat_idx % n_a

    def measure(self, state: ManualLineState):
        return manual_state_to_cam1_pair(state, self.tilt.mask_y_cam1, self.cam_model)

    def mask(self) -> int:
        return int(self.tilt.mask_y_cam1)

    def title(self, idx: int, total: int, state: ManualLineState) -> str:
        pi, ai = self._unpack(idx)
        px_m, py_m = self.tilt.positions[pi]
        deg = math.degrees(self.tilt.alphas_rad[ai])
        extras = (
            f"Move to X={px_m * 100:+.1f}cm Y={py_m * 100:+.1f}cm | "
            f"tilt α_x={deg:+.1f}° | s_px={state.slope_px:+.4f}"
        )
        return _calibration_title("TiltGrid cam1", idx, total, extras, "A/D slope")

    def apply_key(self, state: ManualLineState, key: str, step_x: float, step_s: float) -> None:
        del step_x
        if key == "a":
            state.slope_px += step_s
        elif key == "d":
            state.slope_px -= step_s

    def size(self) -> int:
        return len(self.tilt.positions) * len(self.tilt.alphas_rad)

    def load_sample_into_state(self, state: ManualLineState, idx: int) -> None:
        state.x_at_mask_px = _MIDPOINT_PX
        v = self.tilt.s_px_cam1[idx]
        if v is not None:
            state.slope_px = float(v)

    def save_state_to_sample(self, idx: int, state: ManualLineState) -> None:
        self.tilt.s_px_cam1[idx] = state.slope_px


class TiltGridCam2Stage(Stage):
    """
    Records cam2 slope at each (position, alpha_y) grid point.
    Positions iterate outer, angles iterate inner — move robot once per 5 angles.
    Intercept locked to midpoint; A/D adjust slope only.
    """

    def __init__(self, cam_model: CameraModel, tilt: TiltGrid_Samples) -> None:
        super().__init__("TiltGrid-cam2", "y", cam_index=1, surface_index=1, cam_model=cam_model)
        self.tilt = tilt

    def _unpack(self, flat_idx: int) -> tuple[int, int]:
        n_a = len(self.tilt.alphas_rad)
        return flat_idx // n_a, flat_idx % n_a

    def measure(self, state: ManualLineState):
        return manual_state_to_cam2_pair(state, self.tilt.mask_y_cam2, self.cam_model)

    def mask(self) -> int:
        return int(self.tilt.mask_y_cam2)

    def title(self, idx: int, total: int, state: ManualLineState) -> str:
        pi, ai = self._unpack(idx)
        px_m, py_m = self.tilt.positions[pi]
        deg = math.degrees(self.tilt.alphas_rad[ai])
        extras = (
            f"Move to X={px_m * 100:+.1f}cm Y={py_m * 100:+.1f}cm | "
            f"tilt α_y={deg:+.1f}° | s_px={state.slope_px:+.4f}"
        )
        return _calibration_title("TiltGrid cam2", idx, total, extras, "A/D slope")

    def apply_key(self, state: ManualLineState, key: str, step_x: float, step_s: float) -> None:
        del step_x
        if key == "a":
            state.slope_px += step_s
        elif key == "d":
            state.slope_px -= step_s

    def size(self) -> int:
        return len(self.tilt.positions) * len(self.tilt.alphas_rad)

    def load_sample_into_state(self, state: ManualLineState, idx: int) -> None:
        state.x_at_mask_px = _MIDPOINT_PX
        v = self.tilt.s_px_cam2[idx]
        if v is not None:
            state.slope_px = float(v)

    def save_state_to_sample(self, idx: int, state: ManualLineState) -> None:
        self.tilt.s_px_cam2[idx] = state.slope_px


def _drain_events(reader: DVSReader) -> np.ndarray | None:
    batches = []
    while True:
        b = reader.get_event_batch()
        if b is None or len(b) == 0:
            break
        batches.append(b)
    if not batches:
        return None
    return np.concatenate(batches)


# =========================
# CONTROLLER (sequencing only)
# =========================


class Calibrator:
    def __init__(
        self,
        stages: list[Stage],
        reader_x: DVSReader,
        reader_y: DVSReader,
        viz: OneDvsVisualizer,
        step_x: float,
        step_s: float,
        decay: float,
        display_period: float,
    ) -> None:
        self.stages = stages
        self.reader_x = reader_x
        self.reader_y = reader_y
        self.viz = viz
        self.step_x = step_x
        self.step_s = step_s
        self.decay = decay
        self.display_period = display_period

        self.stage_idx = 0
        self.sample_idx = 0

        self.state = ManualLineState(
            slope_px=0.0,
            x_at_mask_px=DAVIS346_WIDTH / 2,
        )

    def current_stage(self) -> Stage:
        return self.stages[self.stage_idx]

    def current_reader(self) -> DVSReader:
        return self.reader_x if self.current_stage().cam == "x" else self.reader_y

    def save_and_advance(self) -> bool:
        stage = self.current_stage()
        stage.save_state_to_sample(self.sample_idx, self.state)

        self.sample_idx += 1

        if self.sample_idx >= stage.size():
            self.stage_idx += 1
            self.sample_idx = 0

            if self.stage_idx >= len(self.stages):
                return True

        self.load_current()
        return False

    def go_back(self) -> None:
        if self.sample_idx > 0:
            self.sample_idx -= 1
        elif self.stage_idx > 0:
            self.stage_idx -= 1
            self.sample_idx = self.current_stage().size() - 1

        self.load_current()

    def load_current(self) -> None:
        self.current_stage().load_sample_into_state(self.state, self.sample_idx)

    def _clamp_x_at_mask(self, stage: Stage) -> None:
        if isinstance(stage, (XStage, YStage, XYGridCam1Stage, XYGridCam2Stage)):
            self.state.x_at_mask_px = float(max(0.0, min(DAVIS346_WIDTH - 1.0, self.state.x_at_mask_px)))

    def run(self) -> bool:
        W, H = DAVIS346_WIDTH, DAVIS346_HEIGHT
        surface1 = np.zeros((H, W), dtype=np.float32)
        surface2 = np.zeros((H, W), dtype=np.float32)

        def event_frames_fn():
            return surface1, surface2

        self.viz._event_frames_fn = event_frames_fn  # type: ignore[attr-defined]

        next_display = time.perf_counter()
        self.load_current()

        while self.reader_x.is_running() and self.reader_y.is_running():
            stage = self.current_stage()
            reader = self.current_reader()
            my = stage.mask()
            si = stage.surface_index

            ev = _drain_events(reader)
            if ev is not None:
                ev = mask_events_below_line(ev, mask_line_y=my, frame_height=H)
                if si == 0:
                    surface1 *= self.decay
                else:
                    surface2 *= self.decay
                if len(ev) > 0:
                    if si == 0:
                        np.add.at(surface1, (ev["y"], ev["x"]), 1.0)
                    else:
                        np.add.at(surface2, (ev["y"], ev["x"]), 1.0)
            else:
                time.sleep(0.0002)

            now = time.perf_counter()
            if now < next_display:
                continue

            self.viz.cam_index = stage.cam_index

            measurement = stage.measure(self.state)
            title = stage.title(self.sample_idx, stage.size(), self.state)

            vr = self.viz.render(
                measurement,
                title=title,
                mask_line_y=stage.mask(),
            )

            k = vr.key
            if vr.quit or k in (ord("q"), ord("Q"), 27):
                return False

            if k == ord(" "):
                if self.save_and_advance():
                    return True

            elif k == ord("z"):
                self.go_back()

            elif k in (ord("a"), ord("d")):
                stage.apply_key(self.state, chr(k), self.step_x, self.step_s)
                self._clamp_x_at_mask(stage)

            while next_display <= now:
                next_display += self.display_period

        return False


# =========================
# MAIN
# =========================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive simple DVS regression calibrator")
    p.add_argument("--cam1", help="Camera 1 serial/device (omit for discovery)")
    p.add_argument("--cam2", help="Camera 2 serial/device (omit for discovery)")
    p.add_argument("--noise-filter-duration", type=float, default=30.0, metavar="MS", help="Noise filter ms")
    p.add_argument("--mask-y-cam1", type=int, default=int(DEFAULT_CAMERA_PARAMS.y_mask_line_1), metavar="Y", help="Mask line y for cam1")
    p.add_argument("--mask-y-cam2", type=int, default=int(DEFAULT_CAMERA_PARAMS.y_mask_line_2), metavar="Y", help="Mask line y for cam2")
    p.add_argument("--decay-display", type=float, default=0.5, help="Event surface decay")
    p.add_argument("--surface-intensity-gain", type=float, default=50.0, help="Surface brightness (matches OneDvsVisualizer)")
    p.add_argument("--display-fps", type=float, default=30.0, help="GUI refresh rate")
    p.add_argument("--workspace-radius", type=float, default=DEFAULT_WORKSPACE.safe_radius, help="Workspace radius (m)")
    p.add_argument("--x-step-m", type=float, default=0.01, help="B1/B2 grid step along axis (m), used in v1 mode")
    p.add_argument("--port", type=str, default="none", help="Servo port or 'none' for manual move")
    p.add_argument("--settle", type=float, default=2.0, metavar="SEC", help="Settle time after move (seconds)")
    p.add_argument("--tilt-deg-min", type=float, default=-10.0)
    p.add_argument("--tilt-deg-max", type=float, default=10.0)
    p.add_argument("--tilt-deg-step", type=float, default=5.0)
    p.add_argument("--n-position-points", type=int, default=10)
    p.add_argument(
        "--output",
        type=str,
        default=str(default_affine_calibration_path()),
        help="Output path for calibration JSON (default: perception/calibration_files/ next to package)",
    )
    p.add_argument(
        "--dataset-output",
        type=str,
        default="hardware/calibration_files/dvs_calibration_dataset.json",
        help="Combined b1/b2/s1/s2 dataset JSON path",
    )
    p.add_argument("--step-x", type=float, default=1.0, help="B1/B2: A/D step for x_at_mask (pixels)")
    p.add_argument(
        "--step-s",
        type=float,
        default=0.005,
        help="S1/S2: A/D step for slope (px/px)",
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--v2",
        action="store_true",
        default=False,
        help="Use v2 bilinear calibration (2D grid instead of 1D axis sweeps)",
    )
    mode.add_argument(
        "--v3",
        action="store_true",
        default=False,
        help="Use v3 TPS calibration (dense 2D grid + 3D tilt grid)",
    )

    p.add_argument(
        "--grid-step",
        type=float,
        default=0.04,
        metavar="M",
        help="Grid step size in meters for v2 2D position calibration (default: 0.04 = 4cm)",
    )
    p.add_argument(
        "--grid-step-v3",
        type=float,
        default=0.02,
        metavar="M",
        help="v3 position grid step in meters (default: 0.02 = 2cm)",
    )
    p.add_argument(
        "--tilt-positions-v3",
        type=str,
        default="cross",
        choices=["cross", "plus9"],
        help="v3 tilt position pattern: 'cross' (5 pts) or 'plus9' (9 pts)",
    )
    p.add_argument(
        "--tilt-offset-v3",
        type=float,
        default=0.04,
        metavar="M",
        help="Offset of v3 tilt-grid arms from center (default: 0.04 = 4cm)",
    )
    return p.parse_args()


def _init_cams(args):
    if args.cam1 is not None and args.cam2 is not None:
        device1, device2 = args.cam1, args.cam2
    elif args.cam1 is None and args.cam2 is None:
        devices = discover_devices()
        if len(devices) < 2:
            raise SystemExit("Need at least 2 DVS cameras for B1+B2 calibration.")
        device1, device2 = devices[0], devices[1]
    else:
        raise SystemExit("Provide both --cam1 and --cam2, or omit both for discovery.")

    return device1, device2


def main():
    args = parse_args()

    device1, device2 = _init_cams(args)

    reader1 = DVSReader(device1, noise_filter_duration_ms=args.noise_filter_duration)
    reader2 = DVSReader(device2, noise_filter_duration_ms=args.noise_filter_duration)

    viz = OneDvsVisualizer(OneDvsVisualizerParams(
        cam_index=0,
        width=DAVIS346_WIDTH,
        height=DAVIS346_HEIGHT,
        surface_gain=args.surface_intensity_gain,
    ))

    cam_model = CameraModel(width=DAVIS346_WIDTH, height=DAVIS346_HEIGHT)

    # Try to load existing calibration for warm-start predictions.
    existing_model: SimpleDVSRegressionModel | None = None
    try:
        existing_model = SimpleDVSRegressionModel.load(args.output)
        if existing_model.pos_v3 is not None:
            model_desc = "v3 TPS"
        elif existing_model.pos_v2 is not None:
            model_desc = "v2 bilinear"
        else:
            model_desc = "v1 affine"
        print(f"Warm-start: loaded existing {model_desc} model from {args.output}")
    except Exception:
        print(f"No existing calibration found at {args.output} — starting from midpoint")

    tilt_degs = list(DEFAULT_TILT_CALIB_DEGS)
    tilt_rads = tilt_degs_to_rads(tilt_degs)

    if args.v3:
        positions = xy_grid_positions(args.workspace_radius, args.grid_step_v3)
        if len(positions) < 6:
            raise SystemExit(
                f"v3 grid too coarse: {len(positions)} points within radius "
                f"{args.workspace_radius}m at step {args.grid_step_v3}m. Need at least 6."
            )
        print(f"v3 mode: {len(positions)} position points at step={args.grid_step_v3}m")

        d = args.tilt_offset_v3
        if args.tilt_positions_v3 == "cross":
            tilt_positions = [(0.0, 0.0), (+d, 0.0), (-d, 0.0), (0.0, +d), (0.0, -d)]
        else:  # plus9
            tilt_positions = [
                (0.0, 0.0), (+d, 0.0), (-d, 0.0), (0.0, +d), (0.0, -d),
                (+d, +d), (+d, -d), (-d, +d), (-d, -d),
            ]
        print(
            f"v3 tilt: {len(tilt_positions)} positions × {len(tilt_rads)} angles = "
            f"{len(tilt_positions) * len(tilt_rads)} samples per camera"
        )

        b1_init, b2_init = zip(
            *[_predict_pixel_intercepts(existing_model, px, py) for px, py in positions]
        )
        n_ang = len(tilt_rads)
        s1_init: list[float] = []
        s2_init: list[float] = []
        for px, py in tilt_positions:
            for a in tilt_rads:
                s1, s2 = _predict_tilt_slopes(existing_model, a, px_m=px, py_m=py)
                s1_init.append(s1)
                s2_init.append(s2)

        grid = XY_Grid_Samples(
            positions=positions,
            b1_px=list(b1_init),
            b2_px=list(b2_init),
            mask_y_cam1=args.mask_y_cam1,
            mask_y_cam2=args.mask_y_cam2,
        )
        tilt_grid = TiltGrid_Samples(
            positions=tilt_positions,
            alphas_rad=tilt_rads,
            s_px_cam1=s1_init,
            s_px_cam2=s2_init,
            mask_y_cam1=args.mask_y_cam1,
            mask_y_cam2=args.mask_y_cam2,
        )
        stages: list[Stage] = [
            XYGridCam1Stage(cam_model, grid),
            XYGridCam2Stage(cam_model, grid),
            TiltGridCam1Stage(cam_model, tilt_grid),
            TiltGridCam2Stage(cam_model, tilt_grid),
        ]

    elif args.v2:
        positions = xy_grid_positions(args.workspace_radius, args.grid_step)
        if len(positions) < 3:
            raise SystemExit(
                f"Grid too coarse: only {len(positions)} points within radius "
                f"{args.workspace_radius}m at step {args.grid_step}m. Need at least 3."
            )
        print(f"v2 mode: {len(positions)} grid points at step={args.grid_step}m")

        tilt_s_px = [_predict_tilt_slopes(existing_model, r) for r in tilt_rads]
        ax_samples = AX_Samples(ax_pos_rad=list(tilt_rads), s_px=[s1 for s1, _ in tilt_s_px])
        ay_samples = AY_Samples(ay_pos_rad=list(tilt_rads), s_px=[s2 for _, s2 in tilt_s_px])

        b1_init, b2_init = zip(
            *[_predict_pixel_intercepts(existing_model, px, py) for px, py in positions]
        )
        grid = XY_Grid_Samples(
            positions=positions,
            b1_px=list(b1_init),
            b2_px=list(b2_init),
            mask_y_cam1=args.mask_y_cam1,
            mask_y_cam2=args.mask_y_cam2,
        )
        stages = [
            XYGridCam1Stage(cam_model, grid),
            XYGridCam2Stage(cam_model, grid),
            AXStage(cam_model, ax_samples, args.mask_y_cam1),
            AYStage(cam_model, ay_samples, args.mask_y_cam2),
        ]

    else:
        tilt_s_px = [_predict_tilt_slopes(existing_model, r) for r in tilt_rads]
        ax_samples = AX_Samples(ax_pos_rad=list(tilt_rads), s_px=[s1 for s1, _ in tilt_s_px])
        ay_samples = AY_Samples(ay_pos_rad=list(tilt_rads), s_px=[s2 for _, s2 in tilt_s_px])

        init_state = ManualLineState(slope_px=0.0, x_at_mask_px=DAVIS346_WIDTH / 2)
        r = float(args.workspace_radius)
        position_targets = x_positions_from_safe_radius(r, step_m=args.x_step_m)
        samples = build_samples(args, init_state, position_targets, tilt_degs, existing_model)
        stages = [
            XStage("X", "x", 0, 0, cam_model, samples.x),
            YStage("Y", "y", 1, 1, cam_model, samples.y),
            AXStage(cam_model, samples.ax, args.mask_y_cam1),
            AYStage(cam_model, samples.ay, args.mask_y_cam2),
        ]

    calibrator = Calibrator(
        stages=stages,
        reader_x=reader1,
        reader_y=reader2,
        viz=viz,
        step_x=args.step_x,
        step_s=args.step_s,
        decay=args.decay_display,
        display_period=1.0 / args.display_fps,
    )

    try:
        success = calibrator.run()
        if success:
            print("Calibration complete")
            try:
                if args.v3:
                    positions_out, b1, b2, tilt_pos_out, tilt_alphas, s1, s2 = _v3_sample_arrays(
                        grid, tilt_grid
                    )
                    save_tps_v3_calibration(
                        args.output,
                        mask_y_cam1=args.mask_y_cam1,
                        mask_y_cam2=args.mask_y_cam2,
                        positions=positions_out,
                        b1_px=b1,
                        b2_px=b2,
                        tilt_positions=tilt_pos_out,
                        tilt_alphas_rad=tilt_alphas,
                        tilt_s_px_cam1=s1,
                        tilt_s_px_cam2=s2,
                        metadata={
                            "workspace_radius_m": float(args.workspace_radius),
                            "grid_step_m": float(args.grid_step_v3),
                            "tilt_offset_m": float(args.tilt_offset_v3),
                            "tilt_position_pattern": args.tilt_positions_v3,
                            "tilt_grid_deg": list(DEFAULT_TILT_CALIB_DEGS),
                            "source": "simple_dvs_regression_calibrator_v3",
                        },
                    )
                elif args.v2:
                    positions_out, b1, b2, s_ax, axr, s_ay, ayr = _v2_sample_arrays(
                        grid, ax_samples, ay_samples
                    )
                    save_bilinear_v2_calibration(
                        args.output,
                        mask_y_cam1=args.mask_y_cam1,
                        mask_y_cam2=args.mask_y_cam2,
                        positions=positions_out,
                        b1_px=b1,
                        b2_px=b2,
                        slope_px_cam1=s_ax,
                        alpha_x_rad=axr,
                        slope_px_cam2=s_ay,
                        alpha_y_rad=ayr,
                        metadata={
                            "workspace_radius_m": float(args.workspace_radius),
                            "grid_step_m": float(args.grid_step),
                            "n_grid_points": len(positions_out),
                            "tilt_grid_deg": list(DEFAULT_TILT_CALIB_DEGS),
                            "source": "simple_dvs_regression_calibrator_v2",
                        },
                    )
                else:
                    x1, xp, x2, yp, s1, axr, s2, ayr = _affine_sample_arrays(samples)
                    save_affine_v1_calibration(
                        args.output,
                        mask_y_cam1=args.mask_y_cam1,
                        mask_y_cam2=args.mask_y_cam2,
                        x_at_mask_px_cam1=x1,
                        x_pos_m=xp,
                        x_at_mask_px_cam2=x2,
                        y_pos_m=yp,
                        slope_px_cam1=s1,
                        alpha_x_rad=axr,
                        slope_px_cam2=s2,
                        alpha_y_rad=ayr,
                        metadata={
                            "workspace_radius_m": float(args.workspace_radius),
                            "x_step_m": float(args.x_step_m),
                            "tilt_grid_deg": list(DEFAULT_TILT_CALIB_DEGS),
                            "source": "simple_dvs_regression_calibrator",
                        },
                    )
            except ValueError as e:
                raise SystemExit(f"Calibration save failed: {e}") from e
            mode_str = "v3 TPS" if args.v3 else ("v2 bilinear" if args.v2 else "v1 affine")
            print(f"Saved {mode_str} model to {args.output}")
    finally:
        reader1.close()
        reader2.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
