from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np
from src.shared import PoseMeasurement, TableCommand, WorkspaceParams
from visualization.composite_layout import build_composite

from .base import WorkspacePanelRenderer, VizResult, _window_closed
from .sim_dvs import SimDvsVisualizer


@dataclass
class SimDvsWorkspaceVisualizerParams:
    width:  int
    height: int


SIM_DVS_WORKSPACE_VISUALIZER_PRESETS = {
    "default": {
        "width":  346,
        "height": 260,
    }
}


class SimDvsWorkspaceVisualizer(SimDvsVisualizer):
    """Simulated cameras + workspace panel (command dot, grid, pose tilt arrows)."""

    def __init__(
        self,
        workspace: WorkspaceParams,
        width: int = 346,
        height: int = 260,
    ):
        super().__init__(width=width, height=height)
        self._ws = WorkspacePanelRenderer(workspace)

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
        del surfaces, paused
        if measurement is None:
            self._ensure_window(has_workspace=True)
            if _window_closed(self._window_name):
                return VizResult(quit=True)
            key = cv2.waitKey(1) & 0xFF
            return VizResult(quit=key in (ord("q"), ord("Q"), 27))

        self._ensure_window(has_workspace=True)
        frame1, frame2 = self._cam_pair_bgr(measurement)
        workspace_canvas = self._ws.build(command, paused=False, pose=pose)
        title_str = title if title is not None else "Experiment | Q: quit"
        title_str = self._append_pose_banner(title_str, pose)
        composite = build_composite(title_str, frame1, frame2, workspace_canvas)
        cv2.imshow(self._window_name, composite)
        if _window_closed(self._window_name):
            return VizResult(quit=True)
        key = cv2.waitKey(1) & 0xFF
        return VizResult(quit=key in (ord("q"), ord("Q"), 27))
