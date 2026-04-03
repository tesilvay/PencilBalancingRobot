from __future__ import annotations
from dataclasses import dataclass, field

import cv2
import numpy as np
from src.shared import PoseMeasurement, TableCommand, WorkspaceParams, default_workspace
from src.system.sensor.interface.base import get_measurements
from visualization.composite_layout import build_composite

from .base import WorkspacePanelRenderer, EventFramesFn, VizResult, _window_closed
from .real_dvs import RealDvsVisualizer, RealDvsVisualizerParams


@dataclass
class RealDvsWorkspaceVisualizerParams:
    width:           int
    height:          int
    mask_y_cam1:     int
    mask_y_cam2:     int
    workspace:       WorkspaceParams = field(default_factory=default_workspace)
    event_frames_fn: EventFramesFn | None = None


REAL_DVS_WORKSPACE_VISUALIZER_PRESETS = {
    "default": {
        "width":       346,
        "height":      260,
        "mask_y_cam1": 160,
        "mask_y_cam2": 190,
    }
}


class RealDvsWorkspaceVisualizer(RealDvsVisualizer):
    """Real DVS + workspace: pause UI, space toggles pause (returned in VizResult)."""

    def __init__(self, params: RealDvsWorkspaceVisualizerParams):
        super().__init__(RealDvsVisualizerParams(
            width=params.width,
            height=params.height,
            mask_y_cam1=params.mask_y_cam1,
            mask_y_cam2=params.mask_y_cam2,
            event_frames_fn=params.event_frames_fn,
        ))
        self._ws = WorkspacePanelRenderer(params.workspace)

    def render(
        self,
        measurement,
        command: TableCommand | None = None,
        *,
        surfaces: tuple[np.ndarray, np.ndarray] | None = None,
        title: str | None = None,
        paused: bool = False,
        pose: PoseMeasurement | None = None,
    ) -> VizResult:
        del surfaces
        self._ensure_window(has_workspace=True)
        frame1, frame2 = self._bgr_from_surfaces()
        if measurement is not None:
            b1, s1, b2, s2 = get_measurements(measurement)
            if 0 < self.mask_y_cam1 < self.height:
                cv2.line(frame1, (0, self.mask_y_cam1), (self.width - 1, self.mask_y_cam1), (0, 165, 255), 2)
            if 0 < self.mask_y_cam2 < self.height:
                cv2.line(frame2, (0, self.mask_y_cam2), (self.width - 1, self.mask_y_cam2), (0, 165, 255), 2)
            self._draw_line(frame1, b1, s1, mask_y=self.mask_y_cam1)
            self._draw_line(frame2, b2, s2, mask_y=self.mask_y_cam2)

        is_paused = paused is True
        workspace_canvas = self._ws.build(command, paused=is_paused, pose=None if is_paused else pose)
        if title is not None:
            title_str = title
        else:
            title_str = (
                "Paused - table at center | Space: resume | Q: quit"
                if is_paused
                else "Experiment | Space: pause | Q: quit"
            )
        title_str = self._append_pose_banner(title_str, pose)
        composite = build_composite(title_str, frame1, frame2, workspace_canvas)
        cv2.imshow(self._window_name, composite)
        if _window_closed(self._window_name):
            return VizResult(quit=True, toggle_pause=False)
        key = cv2.waitKey(1) & 0xFF
        quit_requested = key in (ord("q"), ord("Q"), 27)
        toggle_pause = key == ord(" ")
        return VizResult(quit=quit_requested, toggle_pause=toggle_pause)
