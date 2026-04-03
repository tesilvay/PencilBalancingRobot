from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np
from src.shared import CameraObservation, PoseMeasurement, TableCommand
from perception.camera_model import CameraModel
from perception.dvs_algorithms import line_x_at_pixel_y
from perception.vision import get_measurements
from visualization.composite_layout import build_one_dvs_composite

from .base import RealtimeVisualizerBase, EventFramesFn, VizResult, _window_closed


@dataclass
class OneDvsVisualizerParams:
    cam_index:    int
    width:        int
    height:       int
    surface_gain: float


ONE_DVS_VISUALIZER_PRESETS = {
    "default": {
        "cam_index":    0,
        "width":        346,
        "height":       260,
        "surface_gain": 50.0,
    }
}


class OneDvsVisualizer(RealtimeVisualizerBase):
    """Single camera panel + line (calibration-oriented)."""

    def __init__(
        self,
        *,
        cam_index: int = 0,
        width: int = 346,
        height: int = 260,
        event_frames_fn: EventFramesFn | None = None,
        surface_gain: float = 50.0,
    ):
        if cam_index not in (0, 1):
            raise ValueError("cam_index must be 0 or 1")
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self.cam = CameraModel(width, height)
        self._event_frames_fn = event_frames_fn
        self._surface_gain = float(surface_gain)
        self._window_ready = False

    def _draw_line_gray(self, img: np.ndarray, b: float, s: float) -> None:
        obs_px = self.cam.camnorm_to_pixel(CameraObservation(slope=s, intercept=b))
        s_px, b_px = obs_px.slope, obs_px.intercept
        s_px = self._to_finite_scalar(s_px)
        b_px = self._to_finite_scalar(b_px)
        if s_px is None or b_px is None:
            return
        y0, y1 = 0, self.height - 1
        x0 = int(round(s_px * y0 + b_px))
        x1 = int(round(s_px * y1 + b_px))
        x0 = max(-10_000, min(10_000, x0))
        x1 = max(-10_000, min(10_000, x1))
        try:
            cv2.line(img, (x0, y0), (x1, y1), 255, 2)
        except cv2.error:
            return

    def _draw_line_masked_bgr(
        self,
        frame: np.ndarray,
        b: float,
        s: float,
        mask_y: int | None,
        *,
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> None:
        obs_px = self.cam.camnorm_to_pixel(CameraObservation(slope=s, intercept=b))
        s_px, b_px = obs_px.slope, obs_px.intercept
        s_px = self._to_finite_scalar(s_px)
        b_px = self._to_finite_scalar(b_px)
        if s_px is None or b_px is None:
            return
        y0 = 0
        if mask_y is not None and 0 < mask_y < self.height:
            y1 = min(mask_y - 1, self.height - 1)
        else:
            y1 = self.height - 1
        x0 = int(round(s_px * y0 + b_px))
        x1 = int(round(s_px * y1 + b_px))
        x0 = max(-10_000, min(10_000, x0))
        x1 = max(-10_000, min(10_000, x1))
        try:
            cv2.line(frame, (x0, y0), (x1, y1), color, 2)
        except cv2.error:
            return
        if mask_y is not None and 0 < mask_y < self.height:
            xi = int(round(line_x_at_pixel_y(obs_px, mask_y)))
            if 0 <= xi < self.width:
                cv2.circle(frame, (xi, mask_y), 5, color, -1)

    def _single_bgr(
        self, measurement, mask_line_y: int | None = None
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        skip_line_for_mask = (
            mask_line_y is not None
            and measurement is not None
            and 0 < int(mask_line_y) < self.height
        )
        if self._event_frames_fn is not None:
            out = self._event_frames_fn()
            if out is not None and len(out) == 2:
                surf = out[self.cam_index]
                bgr = cv2.cvtColor(np.clip(surf * self._surface_gain, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                if measurement is not None and not skip_line_for_mask:
                    b1, s1, b2, s2 = get_measurements(measurement)
                    b, s = (b1, s1) if self.cam_index == 0 else (b2, s2)
                    self._draw_line_bgr_overlay(bgr, b, s)
                return bgr, (0, 255, 0)
        img = np.zeros((self.height, self.width), dtype=np.uint8)
        if measurement is not None and not skip_line_for_mask:
            b1, s1, b2, s2 = get_measurements(measurement)
            b, s = (b1, s1) if self.cam_index == 0 else (b2, s2)
            self._draw_line_gray(img, b, s)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), (255, 255, 255)

    def _draw_line_bgr_overlay(self, frame: np.ndarray, b: float, s: float) -> None:
        obs_px = self.cam.camnorm_to_pixel(CameraObservation(slope=s, intercept=b))
        s_px, b_px = obs_px.slope, obs_px.intercept
        s_px = self._to_finite_scalar(s_px)
        b_px = self._to_finite_scalar(b_px)
        if s_px is None or b_px is None:
            return
        y0, y1 = 0, self.height - 1
        x0 = int(round(s_px * y0 + b_px))
        x1 = int(round(s_px * y1 + b_px))
        x0 = max(-10_000, min(10_000, x0))
        x1 = max(-10_000, min(10_000, x1))
        try:
            cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        except cv2.error:
            return

    def render(
        self,
        measurement,
        command: TableCommand | None = None,
        *,
        surfaces: tuple[np.ndarray, np.ndarray] | None = None,
        title: str | None = None,
        paused: bool = False,
        pose: PoseMeasurement | None = None,
        mask_line_y: int | None = None,
    ) -> VizResult:
        del surfaces, command, paused
        if measurement is None and self._event_frames_fn is None:
            self._ensure_window(has_workspace=False, single_cam=True, one_dvs_side_panel=True)
            if _window_closed(self._window_name):
                return VizResult(quit=True)
            key = cv2.waitKey(1) & 0xFF
            return VizResult(quit=key in (ord("q"), ord("Q"), 27), key=key)

        self._ensure_window(has_workspace=False, single_cam=True, one_dvs_side_panel=True)
        frame1, masked_overlay_color = self._single_bgr(measurement, mask_line_y=mask_line_y)
        if mask_line_y is not None and 0 < int(mask_line_y) < self.height:
            my = int(mask_line_y)
            cv2.line(frame1, (0, my), (self.width - 1, my), (0, 165, 255), 2)
            if measurement is not None:
                b1, s1, b2, s2 = get_measurements(measurement)
                b, s = (b1, s1) if self.cam_index == 0 else (b2, s2)
                self._draw_line_masked_bgr(frame1, b, s, my, color=masked_overlay_color)
        side_text = self._append_pose_banner(
            title if title is not None else "One camera | Q: quit",
            pose,
        )
        composite = build_one_dvs_composite(frame1, side_text, banner_short="One DVS | Q: quit")
        cv2.imshow(self._window_name, composite)
        if _window_closed(self._window_name):
            return VizResult(quit=True)
        key = cv2.waitKey(1) & 0xFF
        return VizResult(quit=key in (ord("q"), ord("Q"), 27), key=key)
