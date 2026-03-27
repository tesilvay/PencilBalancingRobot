"""
S1/S2 slope calibration at table reference (X=0, Y=0): A/D adjusts ``slope_px`` (``s_pix``);
``x_at_mask`` is pinned so the line crosses the mask at the horizontal window center (render
uses only that geometry). Dataset rows are ground-truth angle + ``s*_px`` only.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.sim_types import CameraPair, TableCommand
from perception.camera_model import CameraModel
from perception.dvs_algorithms import mask_events_below_line
from perception.dvs_camera_reader import DAVIS346_HEIGHT, DAVIS346_WIDTH, DVSReader
from visualization.realtime_visualizer import OneDvsVisualizer

from hardware.dvs_calibration.dvs_bx_calibration import ManualLineState, manual_state_to_cam1_pair, manual_state_to_cam2_pair

# Default tilt grid for both cameras (deg), converted to rad at runtime.
DEFAULT_TILT_CALIB_DEGS: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0)


def tilt_degs_to_rads(degs: list[float] | tuple[float, ...]) -> list[float]:
    return [float(np.deg2rad(d)) for d in degs]


@dataclass
class S1Sample:
    """Ground-truth tilt vs measured pixel slope (``ManualLineState.slope_px``)."""

    alpha_x_rad: float
    alpha_x_deg: float
    s1_px: float


@dataclass
class S2Sample:
    alpha_y_rad: float
    alpha_y_deg: float
    s2_px: float


class SxSlopeCalibrationBase(ABC):
    """
    At reference pose: user sets true tilt (prompt in title); A/D adjusts slope only;
    ``x_at_mask_px`` is pinned to image center so lateral line position stays fixed while
    angle changes.
    """

    def __init__(
        self,
        *,
        reader: DVSReader,
        mask_y: int,
        viz: OneDvsVisualizer,
        cam_model: CameraModel,
        angle_targets_rad: list[float],
        decay_display: float,
        display_period: float,
        step_s: float,
        initial_state: ManualLineState,
        device_id: str,
        settle_s: float,
        actuator: Any,
        cam_index: int,
        surface_index: int,
        center_x_at_mask_px: float,
    ) -> None:
        if cam_index not in (0, 1) or surface_index not in (0, 1):
            raise ValueError("cam_index and surface_index must be 0 or 1")
        self.reader = reader
        self.mask_y = int(mask_y)
        self.viz = viz
        self.cam_model = cam_model
        self.angle_targets_rad = angle_targets_rad
        self.decay_display = float(decay_display)
        self.display_period = float(display_period)
        self.step_s = float(step_s)
        self.initial_state = initial_state
        self.device_id = device_id
        self.settle_s = float(settle_s)
        self.actuator = actuator
        self.cam_index = cam_index
        self.surface_index = surface_index
        self.center_x_at_mask_px = float(center_x_at_mask_px)

    @abstractmethod
    def table_command(self, alpha_rad: float) -> TableCommand:
        """Table setpoint (reference); tilt is set by operator to match ``alpha_rad``."""

    @abstractmethod
    def build_measurement(self, state: ManualLineState) -> CameraPair:
        ...

    @abstractmethod
    def record_sample(self, state: ManualLineState, alpha_rad_target: float) -> None:
        ...

    @abstractmethod
    def build_title(
        self,
        *,
        index_1based: int,
        n_total: int,
        alpha_rad_target: float,
        state: ManualLineState,
    ) -> str:
        ...

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

    def _pin_center(self, state: ManualLineState) -> None:
        state.x_at_mask_px = self.center_x_at_mask_px

    def _collect_one_target(
        self,
        *,
        index_1based: int,
        n_total: int,
        alpha_rad_target: float,
    ) -> tuple[ManualLineState | None, bool]:
        W, H = DAVIS346_WIDTH, DAVIS346_HEIGHT
        surface1 = np.zeros((H, W), dtype=np.float32)
        surface2 = np.zeros((H, W), dtype=np.float32)
        state = ManualLineState(
            slope_px=self.initial_state.slope_px,
            x_at_mask_px=self.center_x_at_mask_px,
        )
        self._pin_center(state)
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

            self._pin_center(state)
            measurement = self.build_measurement(state)
            title = self.build_title(
                index_1based=index_1based,
                n_total=n_total,
                alpha_rad_target=alpha_rad_target,
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
                self._pin_center(state)
                return state, False
            if k == ord("a"):
                state.slope_px += self.step_s
            elif k == ord("d"):
                state.slope_px -= self.step_s

            self._pin_center(state)

            while next_display <= now:
                next_display += self.display_period

        return None, True

    def run(self) -> bool:
        n = len(self.angle_targets_rad)
        for i, alpha_rad in enumerate(self.angle_targets_rad):
            ar = float(alpha_rad)
            if self.actuator is not None:
                self.actuator.send(self.table_command(ar))
                time.sleep(self.settle_s)

            st, quit_req = self._collect_one_target(
                index_1based=i + 1,
                n_total=n,
                alpha_rad_target=ar,
            )
            if quit_req or st is None:
                return False
            self._pin_center(st)
            self.initial_state = st
            self.record_sample(st, ar)
        return True


class S1SlopeCalibration(SxSlopeCalibrationBase):
    """Cam1: s1 vs alpha_x at reference; ``cam_index=0``."""

    def __init__(
        self,
        *,
        reader: DVSReader,
        mask_y: int,
        viz: OneDvsVisualizer,
        cam_model: CameraModel,
        angle_targets_rad: list[float],
        decay_display: float,
        display_period: float,
        step_s: float,
        initial_state: ManualLineState,
        device_id: str,
        settle_s: float,
        actuator: Any,
        center_x_at_mask_px: float,
    ) -> None:
        self.samples: list[S1Sample] = []
        super().__init__(
            reader=reader,
            mask_y=mask_y,
            viz=viz,
            cam_model=cam_model,
            angle_targets_rad=angle_targets_rad,
            decay_display=decay_display,
            display_period=display_period,
            step_s=step_s,
            initial_state=initial_state,
            device_id=device_id,
            settle_s=settle_s,
            actuator=actuator,
            cam_index=0,
            surface_index=0,
            center_x_at_mask_px=center_x_at_mask_px,
        )

    def table_command(self, alpha_rad: float) -> TableCommand:
        del alpha_rad
        return TableCommand(x_des=0.0, y_des=0.0)

    def build_measurement(self, state: ManualLineState) -> CameraPair:
        return manual_state_to_cam1_pair(state, self.mask_y, self.cam_model)

    def record_sample(self, state: ManualLineState, alpha_rad_target: float) -> None:
        self.samples.append(
            S1Sample(
                alpha_x_rad=float(alpha_rad_target),
                alpha_x_deg=float(math.degrees(alpha_rad_target)),
                s1_px=float(state.slope_px),
            )
        )

    def build_title(
        self,
        *,
        index_1based: int,
        n_total: int,
        alpha_rad_target: float,
        state: ManualLineState,
    ) -> str:
        tgt_deg = math.degrees(alpha_rad_target)
        s_px = float(state.slope_px)
        return (
            f"S1 cam1 | {index_1based}/{n_total} | set alpha_x={tgt_deg:+.1f} (ref X=Y=0) | "
            f"s1_px={s_px:+.4f} | "
            "A/D slope | SPACE save | Q quit"
        )


class S2SlopeCalibration(SxSlopeCalibrationBase):
    """Cam2: s2 vs alpha_y at reference; ``cam_index=1``."""

    def __init__(
        self,
        *,
        reader: DVSReader,
        mask_y: int,
        viz: OneDvsVisualizer,
        cam_model: CameraModel,
        angle_targets_rad: list[float],
        decay_display: float,
        display_period: float,
        step_s: float,
        initial_state: ManualLineState,
        device_id: str,
        settle_s: float,
        actuator: Any,
        center_x_at_mask_px: float,
    ) -> None:
        self.samples: list[S2Sample] = []
        super().__init__(
            reader=reader,
            mask_y=mask_y,
            viz=viz,
            cam_model=cam_model,
            angle_targets_rad=angle_targets_rad,
            decay_display=decay_display,
            display_period=display_period,
            step_s=step_s,
            initial_state=initial_state,
            device_id=device_id,
            settle_s=settle_s,
            actuator=actuator,
            cam_index=1,
            surface_index=1,
            center_x_at_mask_px=center_x_at_mask_px,
        )

    def table_command(self, alpha_rad: float) -> TableCommand:
        del alpha_rad
        return TableCommand(x_des=0.0, y_des=0.0)

    def build_measurement(self, state: ManualLineState) -> CameraPair:
        return manual_state_to_cam2_pair(state, self.mask_y, self.cam_model)

    def record_sample(self, state: ManualLineState, alpha_rad_target: float) -> None:
        self.samples.append(
            S2Sample(
                alpha_y_rad=float(alpha_rad_target),
                alpha_y_deg=float(math.degrees(alpha_rad_target)),
                s2_px=float(state.slope_px),
            )
        )

    def build_title(
        self,
        *,
        index_1based: int,
        n_total: int,
        alpha_rad_target: float,
        state: ManualLineState,
    ) -> str:
        tgt_deg = math.degrees(alpha_rad_target)
        s_px = float(state.slope_px)
        return (
            f"S2 cam2 | {index_1based}/{n_total} | set alpha_y={tgt_deg:+.1f} (ref X=Y=0) | "
            f"s2_px={s_px:+.4f} | "
            "A/D slope | SPACE save | Q quit"
        )
