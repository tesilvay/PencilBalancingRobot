from __future__ import annotations

import cv2
import numpy as np
from collections.abc import Callable
from dataclasses import dataclass

from core.sim_types import CameraObservation, PoseMeasurement, TableCommand, WorkspaceParams
from perception.camera_model import CameraModel
from perception.dvs_algorithms import line_x_at_pixel_y
from perception.vision import get_measurements
from visualization.composite_layout import build_composite, build_one_dvs_composite, get_default_window_size

EventFramesFn = Callable[[], tuple[np.ndarray, np.ndarray] | None]

_TILT_ARROW_CAP_RAD = float(np.deg2rad(15.0))
DEFAULT_TILT_ARROW_MAX_LENGTH_PX = 40.0


def _window_closed(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


@dataclass(frozen=True)
class VizResult:
    quit: bool
    toggle_pause: bool = False
    key: int | None = None


class WorkspacePanelRenderer:
    """Workspace grid, safe circle, command dot, paused slate, optional tilt arrow from pose."""

    def __init__(
        self,
        workspace: WorkspaceParams,
        *,
        tilt_arrow_max_length_px: float = DEFAULT_TILT_ARROW_MAX_LENGTH_PX,
    ):
        self.workspace = workspace
        self.tilt_arrow_max_length_px = float(tilt_arrow_max_length_px)
        self._workspace_size = 350
        self._center = self._workspace_size // 2
        self._workspace_margin = 20
        self._grid_step_m = 0.02
        if workspace.safe_radius is not None:
            self._scale = (self._workspace_size - 2 * self._workspace_margin) / (2 * workspace.safe_radius)
        else:
            self._scale = 4000.0

    def _draw_tilt_arrow(self, canvas: np.ndarray, px: int, py: int, pose: PoseMeasurement) -> None:
        ax = float(pose.alpha_x)
        ay = float(pose.alpha_y)
        m = float(np.hypot(ax, ay))
        if not np.isfinite(m) or m < 1e-9:
            return
        ux, uy_world = ax / m, ay / m
        ux_img, uy_img = ux, -uy_world
        m_eff = min(m, _TILT_ARROW_CAP_RAD)
        length = (m_eff / _TILT_ARROW_CAP_RAD) * self.tilt_arrow_max_length_px
        ex = int(round(px + length * ux_img))
        ey = int(round(py + length * uy_img))
        cv2.circle(canvas, (px, py), 3, (255, 255, 255), -1)
        cv2.arrowedLine(canvas, (px, py), (ex, ey), (80, 200, 255), 2, tipLength=0.3)
        cv2.circle(canvas, (ex, ey), 3, (80, 200, 255), -1)

    def build(self, command: TableCommand | None, *, paused: bool = False, pose: PoseMeasurement | None = None) -> np.ndarray:
        if paused:
            canvas = np.zeros((self._workspace_size, self._workspace_size), dtype=np.uint8)
            canvas[:] = 30
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            text = "Paused - table at center"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            tx = (self._workspace_size - tw) // 2
            ty = (self._workspace_size + th) // 2
            cv2.putText(canvas, text, (tx, ty), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(canvas, text, (tx, ty), font, font_scale, (255, 255, 255), thickness)
            return canvas

        canvas = np.zeros((self._workspace_size, self._workspace_size), dtype=np.uint8)
        canvas[:] = 40
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        x_ref = self.workspace.x_ref
        y_ref = self.workspace.y_ref
        safe_radius = self.workspace.safe_radius
        grid_color = (55, 55, 55)
        circle_color = (100, 100, 100)

        if safe_radius is not None:
            n_grid = int(np.ceil(safe_radius / self._grid_step_m))
            for k in range(-n_grid, n_grid + 1):
                x_world = x_ref + k * self._grid_step_m
                px = int(self._center + (x_world - x_ref) * self._scale)
                if 0 <= px < self._workspace_size:
                    cv2.line(canvas, (px, 0), (px, self._workspace_size - 1), grid_color, 1)
                y_world = y_ref + k * self._grid_step_m
                py = int(self._center - (y_world - y_ref) * self._scale)
                if 0 <= py < self._workspace_size:
                    cv2.line(canvas, (0, py), (self._workspace_size - 1, py), grid_color, 1)
            radius_px = int(safe_radius * self._scale)
            cv2.circle(canvas, (self._center, self._center), radius_px, circle_color, 1)

        cross_len = 15
        cv2.line(
            canvas,
            (self._center - cross_len, self._center),
            (self._center + cross_len, self._center),
            circle_color,
            1,
        )
        cv2.line(
            canvas,
            (self._center, self._center - cross_len),
            (self._center, self._center + cross_len),
            circle_color,
            1,
        )

        if command is not None:
            x_des, y_des = command.x_des, command.y_des
            px = int(self._center + (x_des - x_ref) * self._scale)
            py = int(self._center - (y_des - y_ref) * self._scale)
            if 0 <= px < self._workspace_size and 0 <= py < self._workspace_size:
                cv2.circle(canvas, (px, py), 5, (0, 255, 0), -1)

        if pose is not None:
            px_p = int(self._center + (pose.X - x_ref) * self._scale)
            py_p = int(self._center - (pose.Y - y_ref) * self._scale)
            if 0 <= px_p < self._workspace_size and 0 <= py_p < self._workspace_size:
                self._draw_tilt_arrow(canvas, px_p, py_p, pose)

        return canvas


class RealtimeVisualizerBase:
    _window_name = "Pencil Balancer"

    @staticmethod
    def _to_finite_scalar(x) -> float | None:
        if isinstance(x, np.ndarray):
            if x.size != 1:
                return None
            x = x.reshape(-1)[0]
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(xf):
            return None
        return xf

    def _ensure_window(
        self,
        *,
        has_workspace: bool,
        single_cam: bool = False,
        one_dvs_side_panel: bool = False,
    ) -> None:
        if getattr(self, "_window_ready", False):
            return
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        w, h = get_default_window_size(
            has_cams=True,
            has_workspace=has_workspace,
            single_cam=single_cam,
            one_dvs_side_panel=one_dvs_side_panel,
        )
        cv2.resizeWindow(self._window_name, w, h)
        self._window_ready = True

    @staticmethod
    def _append_pose_banner(title_str: str, pose: PoseMeasurement | None) -> str:
        if pose is None:
            return title_str
        x_mm = pose.X * 1000.0
        y_mm = pose.Y * 1000.0
        ax_deg = pose.alpha_x * 180.0 / np.pi
        ay_deg = pose.alpha_y * 180.0 / np.pi
        return (
            title_str
            + f" | X={x_mm:6.1f} mm, Y={y_mm:6.1f} mm, ax={ax_deg:5.1f} deg, ay={ay_deg:5.1f} deg"
        )
