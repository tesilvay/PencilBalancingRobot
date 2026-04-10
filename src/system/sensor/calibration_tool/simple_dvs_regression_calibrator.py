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
)
from src.experiment.realtime_visualizer import OneDvsVisualizer, OneDvsVisualizerParams




def x_positions_from_safe_radius(safe_radius_m: float, step_m: float = 0.01) -> list[float]:
    r = float(safe_radius_m)
    step = float(step_m)
    if step <= 0 or not math.isfinite(step):
        raise ValueError("step_m must be positive and finite")
    n = int(math.floor((r - 1e-9) / step))
    if n < 0:
        n = 0
    return [k * step for k in range(-n, n + 1)]


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
class Samples:
    x: X_Samples
    y: Y_Samples
    ax: AX_Samples
    ay: AY_Samples


def build_samples(args, init_state, position_targets, tilt_degs):
    tilt_rads = tilt_degs_to_rads(tilt_degs)

    n_pos = len(position_targets)
    n_tilt = len(tilt_rads)

    return Samples(
        x=X_Samples(
            x_pos_m=list(position_targets),
            x_at_mask_px=[init_state.x_at_mask_px] + [None] * (n_pos - 1),
            mask_y=args.mask_y_cam1,
        ),
        y=Y_Samples(
            y_pos_m=list(position_targets),
            x_at_mask_px=[init_state.x_at_mask_px] + [None] * (n_pos - 1),
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
        ang = math.degrees(math.atan(obs.slope))
        x = self.samples.x_pos_m[idx]
        extras = (
            f"target X={x * 100:+.1f} cm | deg={ang:+.2f} | "
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
        ang = math.degrees(math.atan(obs.slope))
        y = self.samples.y_pos_m[idx]
        extras = (
            f"target Y={y * 100:+.1f} cm | deg={ang:+.2f} | "
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
        v = self.samples.x_at_mask_px[idx]
        if v is not None:
            state.x_at_mask_px = float(v)

    def save_state_to_sample(self, idx: int, state: ManualLineState) -> None:
        self.samples.x_at_mask_px[idx] = state.x_at_mask_px


class AXStage(Stage):
    """AX on camera 1; mask y matches X stage (events below mask discarded)."""

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
        v = self.samples.s_px[idx]
        if v is not None:
            state.slope_px = float(v)

    def save_state_to_sample(self, idx: int, state: ManualLineState) -> None:
        self.samples.s_px[idx] = state.slope_px


class AYStage(Stage):
    """AY on camera 2."""

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
        v = self.samples.s_px[idx]
        if v is not None:
            state.slope_px = float(v)

    def save_state_to_sample(self, idx: int, state: ManualLineState) -> None:
        self.samples.s_px[idx] = state.slope_px


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
        if isinstance(stage, (XStage, YStage)):
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
    p.add_argument("--x-step-m", type=float, default=0.01, help="B1/B2 grid step along axis (m)")
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
        help="Output path for simple_dvs_regression_v1 affine JSON (default: perception/calibration_files/ next to package)",
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
        help="S1/S2: A/D step for slope (px/px); legacy full regression uses W/S with same step",
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

    init_state = ManualLineState(
        slope_px=0.0,
        x_at_mask_px=DAVIS346_WIDTH / 2,
    )

    r = float(args.workspace_radius)
    position_targets = x_positions_from_safe_radius(r, step_m=args.x_step_m)
    tilt_degs = list(DEFAULT_TILT_CALIB_DEGS)

    samples = build_samples(args, init_state, position_targets, tilt_degs)

    stages: list[Stage] = [
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
                raise SystemExit(f"Affine save failed: {e}") from e
            print(f"Saved affine model to {args.output}")
    finally:
        reader1.close()
        reader2.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
