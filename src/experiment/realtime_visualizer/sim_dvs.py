from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np
from src.shared import CameraObservation, Measurement, ControlInput
from src.system.sensor.observation_model.camera_model import CameraModel
from .multi_panel_layout import build_composite

from .base import RealtimeVisualizerBase, VizResult, _window_closed


@dataclass
class SimDvsVisualizerParams:
    width:  int
    height: int


SIM_DVS_VISUALIZER_PRESETS = {
    "default": {
        "width":  346,
        "height": 260,
    }
}


class SimDvsVisualizer(RealtimeVisualizerBase):
    """Simulated gray camera pair + line overlay (no workspace)."""

    def __init__(self, params: SimDvsVisualizerParams):
        self.width = params.width
        self.height = params.height
        self.cam = CameraModel(params.width, params.height)
        self._window_ready = False

    def draw_line(self, img: np.ndarray, b: float, s: float) -> None:
        obs_px = self.cam.camnorm_to_pixel(CameraObservation(slope=s, intercept=b))
        s_px, b_px = obs_px.slope, obs_px.intercept

        s_px = self._to_finite_scalar(s_px)
        b_px = self._to_finite_scalar(b_px)
        if s_px is None or b_px is None:
            return

        y0 = 0
        y1 = self.height - 1
        x0 = int(round(s_px * y0 + b_px))
        x1 = int(round(s_px * y1 + b_px))
        x0 = max(-10_000, min(10_000, x0))
        x1 = max(-10_000, min(10_000, x1))
        try:
            cv2.line(img, (x0, y0), (x1, y1), 255, 2)
        except cv2.error:
            return

    def _cam_pair_bgr(self, measurement) -> tuple[np.ndarray, np.ndarray]:
        img1 = np.zeros((self.height, self.width), dtype=np.uint8)
        img2 = np.zeros((self.height, self.width), dtype=np.uint8)
        if measurement is not None:
            b1, s1, b2, s2 = measurement.unpack()
            self.draw_line(img1, b1, s1)
            self.draw_line(img2, b2, s2)
        f1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        f2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        return f1, f2

    def render(
        self,
        measurement,
        command: ControlInput | None = None,
        *,
        surfaces: tuple[np.ndarray, np.ndarray] | None = None,
        title: str | None = None,
        paused: bool = False,
        y_meas: Measurement | None = None,
    ) -> VizResult:
        del surfaces, command, paused
        if measurement is None:
            self._ensure_window(has_workspace=False)
            if _window_closed(self._window_name):
                return VizResult(quit=True)
            key = cv2.waitKey(1) & 0xFF
            return VizResult(quit=key in (ord("q"), ord("Q"), 27))

        self._ensure_window(has_workspace=False)
        frame1, frame2 = self._cam_pair_bgr(measurement)
        title_str = title if title is not None else "Experiment | Q: quit"
        title_str = self._append_y_meas_banner(title_str, y_meas)
        composite = build_composite(title_str, frame1, frame2, None)
        cv2.imshow(self._window_name, composite)
        if _window_closed(self._window_name):
            return VizResult(quit=True)
        key = cv2.waitKey(1) & 0xFF
        return VizResult(quit=key in (ord("q"), ord("Q"), 27))
