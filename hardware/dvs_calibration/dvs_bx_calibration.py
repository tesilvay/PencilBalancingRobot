"""
B1/B2 intercept calibration: shared UI loop (events, mask, A/D line, OneDvsVisualizer) with
subclasses holding only stage-specific data and wiring (camera index, table command, samples).

S1/S2 slope calibration lives in :mod:`hardware.dvs_sx_calibration`.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.sim_types import CameraObservation, CameraPair, TableCommand
from perception.camera_model import CameraModel
from perception.dvs_algorithms import mask_events_below_line
from perception.dvs_camera_reader import DAVIS346_HEIGHT, DAVIS346_WIDTH, DVSReader
from visualization.realtime_visualizer import OneDvsVisualizer


@dataclass
class ManualLineState:
    slope_px: float = 0.0
    x_at_mask_px: float = 0.0

    def to_obs_px(self, mask_y: int) -> tuple[float, float]:
        b_px = float(self.x_at_mask_px) - float(self.slope_px) * float(mask_y)
        return float(self.slope_px), float(b_px)


def line_angle_deg_from_slope_px(slope_px: float) -> float:
    return float(math.degrees(math.atan(float(slope_px))))


def manual_state_to_cam1_pair(state: ManualLineState, mask_y: int, cam: CameraModel) -> CameraPair:
    s_px, b_px = state.to_obs_px(mask_y=mask_y)
    n = cam.pixel_to_camnorm(CameraObservation(slope=s_px, intercept=b_px))
    return CameraPair(cam1=n, cam2=CameraObservation(slope=0.0, intercept=0.0))


def manual_state_to_cam2_pair(state: ManualLineState, mask_y: int, cam: CameraModel) -> CameraPair:
    s_px, b_px = state.to_obs_px(mask_y=mask_y)
    n = cam.pixel_to_camnorm(CameraObservation(slope=s_px, intercept=b_px))
    return CameraPair(cam1=CameraObservation(slope=0.0, intercept=0.0), cam2=n)


def x_positions_from_safe_radius(safe_radius_m: float, step_m: float = 0.01) -> list[float]:
    r = float(safe_radius_m)
    step = float(step_m)
    if step <= 0 or not math.isfinite(step):
        raise ValueError("step_m must be positive and finite")
    n = int(math.floor((r - 1e-9) / step))
    if n < 0:
        n = 0
    return [k * step for k in range(-n, n + 1)]


@dataclass
class B1Sample:
    x_pos_m: float
    b1_px: float
    b1_camnorm: float
    s1_px: float
    x_at_mask_px: float
    mask_y: int


@dataclass
class B2Sample:
    y_pos_m: float
    b2_px: float
    b2_camnorm: float
    s2_px: float
    x_at_mask_px: float
    mask_y: int


class BxInterceptCalibrationBase(ABC):
    """
    Shared behavior: drain events into one of two surfaces, render selected cam via
    ``OneDvsVisualizer``, A/D horizontal line only, Space to save, Q to quit.

    Subclasses supply table commands, ``CameraPair`` construction, titles, and append to
    their own sample lists.
    """

    def __init__(
        self,
        *,
        reader: DVSReader,
        mask_y: int,
        viz: OneDvsVisualizer,
        cam_model: CameraModel,
        position_targets_m: list[float],
        decay_display: float,
        display_period: float,
        step_x: float,
        initial_state: ManualLineState,
        device_id: str,
        safe_radius_m: float,
        x_step_m: float,
        settle_s: float,
        actuator: Any,
        cam_index: int,
        surface_index: int,
    ) -> None:
        if cam_index not in (0, 1) or surface_index not in (0, 1):
            raise ValueError("cam_index and surface_index must be 0 or 1")
        self.reader = reader
        self.mask_y = int(mask_y)
        self.viz = viz
        self.cam_model = cam_model
        self.position_targets_m = position_targets_m
        self.decay_display = float(decay_display)
        self.display_period = float(display_period)
        self.step_x = float(step_x)
        self.initial_state = initial_state
        self.device_id = device_id
        self.safe_radius_m = float(safe_radius_m)
        self.x_step_m = float(x_step_m)
        self.settle_s = float(settle_s)
        self.actuator = actuator
        self.cam_index = cam_index
        self.surface_index = surface_index

    @abstractmethod
    def table_command(self, target_m: float) -> TableCommand:
        """Table setpoint for this target (B1: move X; B2: move Y)."""

    @abstractmethod
    def build_measurement(self, state: ManualLineState) -> CameraPair:
        """Pixel manual line -> normalized pair for the active camera."""

    @abstractmethod
    def record_sample(self, state: ManualLineState, target_m: float) -> None:
        """Append one calibration row to the subclass-owned storage."""

    @abstractmethod
    def build_title(
        self,
        *,
        index_1based: int,
        n_total: int,
        target_m: float,
        state: ManualLineState,
    ) -> str:
        """Banner text for the current target (cm / deg / b_px shown here)."""

    def _drain_events(self) -> np.ndarray | None:
        batches = []
        while True:
            b = self.reader.get_event_batch()
            if b is None or len(b) == 0:
                break
            batches.append(b)
        if not batches:
            return None
        return np.concatenate(batches)

    def _collect_one_target(
        self,
        *,
        index_1based: int,
        n_total: int,
        target_m: float,
    ) -> tuple[ManualLineState | None, bool]:
        W, H = DAVIS346_WIDTH, DAVIS346_HEIGHT
        surface1 = np.zeros((H, W), dtype=np.float32)
        surface2 = np.zeros((H, W), dtype=np.float32)
        state = ManualLineState(
            slope_px=self.initial_state.slope_px,
            x_at_mask_px=self.initial_state.x_at_mask_px,
        )
        my = int(self.mask_y)
        si = self.surface_index

        def event_frames_fn():
            return surface1, surface2

        self.viz._event_frames_fn = event_frames_fn  # type: ignore[attr-defined]
        self.viz.cam_index = self.cam_index

        next_display = time.perf_counter()
        while self.reader.is_running():
            ev = self._drain_events()
            if ev is not None:
                ev = mask_events_below_line(ev, mask_line_y=my, frame_height=H)
                if si == 0:
                    surface1 *= self.decay_display
                else:
                    surface2 *= self.decay_display
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

            measurement = self.build_measurement(state)
            title = self.build_title(
                index_1based=index_1based,
                n_total=n_total,
                target_m=target_m,
                state=state,
            )
            vr = self.viz.render(
                measurement,
                title=title,
                mask_line_y=my,
            )
            if vr.quit:
                return None, True
            k = vr.key
            if k in (ord("q"), ord("Q"), 27):
                return None, True
            if k == ord(" "):
                return state, False
            if k == ord("a"):
                state.x_at_mask_px -= self.step_x
            elif k == ord("d"):
                state.x_at_mask_px += self.step_x

            state.x_at_mask_px = float(max(0.0, min(W - 1.0, state.x_at_mask_px)))

            while next_display <= now:
                next_display += self.display_period

        return None, True

    def run(self) -> bool:
        n = len(self.position_targets_m)
        for i, pos_m in enumerate(self.position_targets_m):
            if self.actuator is not None:
                self.actuator.send(self.table_command(float(pos_m)))
                time.sleep(self.settle_s)

            st, quit_req = self._collect_one_target(
                index_1based=i + 1,
                n_total=n,
                target_m=float(pos_m),
            )
            if quit_req or st is None:
                return False
            self.initial_state = st
            self.record_sample(st, float(pos_m))
        return True


class B1InterceptCalibration(BxInterceptCalibrationBase):
    """Cam1: b1 vs global X; uses ``cam_index=0`` and reader1."""

    def __init__(
        self,
        *,
        reader: DVSReader,
        mask_y: int,
        viz: OneDvsVisualizer,
        cam_model: CameraModel,
        position_targets_m: list[float],
        decay_display: float,
        display_period: float,
        step_x: float,
        initial_state: ManualLineState,
        device_id: str,
        safe_radius_m: float,
        x_step_m: float,
        settle_s: float,
        actuator: Any,
    ) -> None:
        self.samples: list[B1Sample] = []
        super().__init__(
            reader=reader,
            mask_y=mask_y,
            viz=viz,
            cam_model=cam_model,
            position_targets_m=position_targets_m,
            decay_display=decay_display,
            display_period=display_period,
            step_x=step_x,
            initial_state=initial_state,
            device_id=device_id,
            safe_radius_m=safe_radius_m,
            x_step_m=x_step_m,
            settle_s=settle_s,
            actuator=actuator,
            cam_index=0,
            surface_index=0,
        )

    def table_command(self, target_m: float) -> TableCommand:
        return TableCommand(x_des=float(target_m), y_des=0.0)

    def build_measurement(self, state: ManualLineState) -> CameraPair:
        return manual_state_to_cam1_pair(state, self.mask_y, self.cam_model)

    def record_sample(self, state: ManualLineState, target_m: float) -> None:
        s_px, b_px = state.to_obs_px(mask_y=int(self.mask_y))
        n_norm = self.cam_model.pixel_to_camnorm(CameraObservation(slope=s_px, intercept=b_px))
        self.samples.append(
            B1Sample(
                x_pos_m=float(target_m),
                b1_px=float(b_px),
                b1_camnorm=float(n_norm.intercept),
                s1_px=float(s_px),
                x_at_mask_px=float(state.x_at_mask_px),
                mask_y=int(self.mask_y),
            )
        )

    def build_title(
        self,
        *,
        index_1based: int,
        n_total: int,
        target_m: float,
        state: ManualLineState,
    ) -> str:
        s_px, b_px = state.to_obs_px(mask_y=int(self.mask_y))
        ang_deg = line_angle_deg_from_slope_px(s_px)
        return (
            f"B1 cam1 | {index_1based}/{n_total} | target X={target_m * 100:+.1f} cm, Y=0 | "
            f"line deg={ang_deg:+.2f} | b1_px={b_px:+.2f} | x@mask={float(state.x_at_mask_px):+.1f} | "
            "A/D move | SPACE save | Q quit"
        )


class B2InterceptCalibration(BxInterceptCalibrationBase):
    """Cam2: b2 vs global Y; uses ``cam_index=1`` and reader2."""

    def __init__(
        self,
        *,
        reader: DVSReader,
        mask_y: int,
        viz: OneDvsVisualizer,
        cam_model: CameraModel,
        position_targets_m: list[float],
        decay_display: float,
        display_period: float,
        step_x: float,
        initial_state: ManualLineState,
        device_id: str,
        safe_radius_m: float,
        x_step_m: float,
        settle_s: float,
        actuator: Any,
    ) -> None:
        self.samples: list[B2Sample] = []
        super().__init__(
            reader=reader,
            mask_y=mask_y,
            viz=viz,
            cam_model=cam_model,
            position_targets_m=position_targets_m,
            decay_display=decay_display,
            display_period=display_period,
            step_x=step_x,
            initial_state=initial_state,
            device_id=device_id,
            safe_radius_m=safe_radius_m,
            x_step_m=x_step_m,
            settle_s=settle_s,
            actuator=actuator,
            cam_index=1,
            surface_index=1,
        )

    def table_command(self, target_m: float) -> TableCommand:
        return TableCommand(x_des=0.0, y_des=float(target_m))

    def build_measurement(self, state: ManualLineState) -> CameraPair:
        return manual_state_to_cam2_pair(state, self.mask_y, self.cam_model)

    def record_sample(self, state: ManualLineState, target_m: float) -> None:
        s_px, b_px = state.to_obs_px(mask_y=int(self.mask_y))
        n_norm = self.cam_model.pixel_to_camnorm(CameraObservation(slope=s_px, intercept=b_px))
        self.samples.append(
            B2Sample(
                y_pos_m=float(target_m),
                b2_px=float(b_px),
                b2_camnorm=float(n_norm.intercept),
                s2_px=float(s_px),
                x_at_mask_px=float(state.x_at_mask_px),
                mask_y=int(self.mask_y),
            )
        )

    def build_title(
        self,
        *,
        index_1based: int,
        n_total: int,
        target_m: float,
        state: ManualLineState,
    ) -> str:
        s_px, b_px = state.to_obs_px(mask_y=int(self.mask_y))
        ang_deg = line_angle_deg_from_slope_px(s_px)
        return (
            f"B2 cam2 | {index_1based}/{n_total} | target X=0, Y={target_m * 100:+.1f} cm | "
            f"line deg={ang_deg:+.2f} | b2_px={b_px:+.2f} | x@mask={float(state.x_at_mask_px):+.1f} | "
            "A/D move | SPACE save | Q quit"
        )
