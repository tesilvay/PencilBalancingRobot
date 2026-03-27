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

# Tilt arrow saturates at this combined angle (|| (α_x, α_y) || in radians).
_TILT_ARROW_CAP_RAD = float(np.deg2rad(15.0))
# Pixel length at that cap (tune here or pass ``tilt_arrow_max_length_px=`` to WorkspacePanelRenderer).
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
    # Last key from cv2.waitKey (0–255) when the visualizer consumed input; None if unused.
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
        """Single arrow: +α_x → right; length scales with tilt up to max at 15° (see module cap)."""
        ax = float(pose.alpha_x)
        ay = float(pose.alpha_y)
        m = float(np.hypot(ax, ay))
        if not np.isfinite(m) or m < 1e-9:
            return
        # Direction in workspace: x right, y up on canvas (world +Y is up → image -y).
        ux, uy_world = ax / m, ay / m
        ux_img, uy_img = ux, -uy_world
        m_eff = min(m, _TILT_ARROW_CAP_RAD)
        length = (m_eff / _TILT_ARROW_CAP_RAD) * self.tilt_arrow_max_length_px
        ex = int(round(px + length * ux_img))
        ey = int(round(py + length * uy_img))
        # Base marker improves readability when arrow is short.
        cv2.circle(canvas, (px, py), 3, (255, 255, 255), -1)
        cv2.arrowedLine(canvas, (px, py), (ex, ey), (80, 200, 255), 2, tipLength=0.3)
        # Tip marker makes arrow endpoint easier to identify.
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


class SimDvsVisualizer(RealtimeVisualizerBase):
    """Simulated gray camera pair + line overlay (no workspace)."""

    def __init__(self, width: int = 346, height: int = 260):
        self.width = width
        self.height = height
        self.cam = CameraModel(width, height)
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
            b1, s1, b2, s2 = get_measurements(measurement)
            self.draw_line(img1, b1, s1)
            self.draw_line(img2, b2, s2)
        f1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        f2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        return f1, f2

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
        title_str = self._append_pose_banner(title_str, pose)
        composite = build_composite(title_str, frame1, frame2, None)
        cv2.imshow(self._window_name, composite)
        if _window_closed(self._window_name):
            return VizResult(quit=True)
        key = cv2.waitKey(1) & 0xFF
        return VizResult(quit=key in (ord("q"), ord("Q"), 27))


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


class RealDvsVisualizer(RealtimeVisualizerBase):
    """Real DVS event-accumulator panels + line and mask overlays."""

    def __init__(
        self,
        event_frames_fn: EventFramesFn | None,
        width: int = 346,
        height: int = 260,
        mask_y_cam1: int = 160,
        mask_y_cam2: int = 190,
    ):
        self._event_frames_fn = event_frames_fn
        self.width = width
        self.height = height
        self.cam = CameraModel(width, height)
        self.mask_y_cam1 = int(mask_y_cam1)
        self.mask_y_cam2 = int(mask_y_cam2)
        self._window_ready = False

    def _bgr_from_surfaces(self) -> tuple[np.ndarray, np.ndarray]:
        if self._event_frames_fn is None:
            z = np.zeros((self.height, self.width), dtype=np.uint8)
            bgr = cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)
            return bgr, bgr
        out = self._event_frames_fn()
        if out is None or len(out) != 2:
            z = np.zeros((self.height, self.width), dtype=np.uint8)
            bgr = cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)
            return bgr, bgr
        surface1, surface2 = out
        frame1 = np.clip(surface1 * 50, 0, 255).astype(np.uint8)
        frame2 = np.clip(surface2 * 50, 0, 255).astype(np.uint8)
        return cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR), cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)

    def _draw_line(self, frame: np.ndarray, b: float, s: float, mask_y: int | None = None) -> None:
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
            cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        except cv2.error:
            return
        if mask_y is not None and 0 < mask_y < self.height:
            xi = int(round(line_x_at_pixel_y(obs_px, mask_y)))
            if 0 <= xi < self.width:
                cv2.circle(frame, (xi, mask_y), 5, (0, 255, 0), -1)

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
        del surfaces, command, paused
        self._ensure_window(has_workspace=False)
        frame1, frame2 = self._bgr_from_surfaces()
        if measurement is not None:
            b1, s1, b2, s2 = get_measurements(measurement)
            if 0 < self.mask_y_cam1 < self.height:
                cv2.line(frame1, (0, self.mask_y_cam1), (self.width - 1, self.mask_y_cam1), (0, 165, 255), 2)
            if 0 < self.mask_y_cam2 < self.height:
                cv2.line(frame2, (0, self.mask_y_cam2), (self.width - 1, self.mask_y_cam2), (0, 165, 255), 2)
            self._draw_line(frame1, b1, s1, mask_y=self.mask_y_cam1)
            self._draw_line(frame2, b2, s2, mask_y=self.mask_y_cam2)

        title_str = title if title is not None else "Experiment | Q: quit"
        title_str = self._append_pose_banner(title_str, pose)
        composite = build_composite(title_str, frame1, frame2, None)
        cv2.imshow(self._window_name, composite)
        if _window_closed(self._window_name):
            return VizResult(quit=True)
        key = cv2.waitKey(1) & 0xFF
        return VizResult(quit=key in (ord("q"), ord("Q"), 27))


class RealDvsWorkspaceVisualizer(RealDvsVisualizer):
    """Real DVS + workspace: pause UI, space toggles pause (returned in VizResult)."""

    def __init__(
        self,
        workspace: WorkspaceParams,
        event_frames_fn: EventFramesFn | None,
        width: int = 346,
        height: int = 260,
        mask_y_cam1: int = 160,
        mask_y_cam2: int = 190,
    ):
        super().__init__(event_frames_fn, width=width, height=height, mask_y_cam1=mask_y_cam1, mask_y_cam2=mask_y_cam2)
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
        """Line from top to just above ``mask_y``, plus a dot at the line–mask intersection (BGR)."""
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
        """Returns BGR frame and color for mask-clipped line overlay (green on DVS surface, white on sim gray)."""
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

#deprecated

class PencilVisualizerRealtime:
    """Deprecated: use :class:`SimDvsVisualizer` or :class:`SimDvsWorkspaceVisualizer`."""

    def __init__(
        self,
        width: int = 346,
        height: int = 260,
        show_workspace: bool = False,
        workspace: WorkspaceParams | None = None,
    ):
        if show_workspace and workspace is not None:
            self._impl: SimDvsVisualizer = SimDvsWorkspaceVisualizer(workspace, width=width, height=height)
        else:
            self._impl = SimDvsVisualizer(width=width, height=height)

    def render(self, *args, **kwargs) -> VizResult:
        return self._impl.render(*args, **kwargs)


class DVSWorkspaceVisualizer:
    """Deprecated: use :class:`RealDvsVisualizer` or :class:`RealDvsWorkspaceVisualizer`."""

    def __init__(
        self,
        workspace: WorkspaceParams,
        width: int = 346,
        height: int = 260,
        show_workspace: bool = True,
        mask_y_cam1: int = 160,
        mask_y_cam2: int = 190,
        event_frames_fn: EventFramesFn | None = None,
    ):
        if show_workspace:
            self._impl: RealDvsVisualizer = RealDvsWorkspaceVisualizer(
                workspace,
                event_frames_fn,
                width=width,
                height=height,
                mask_y_cam1=mask_y_cam1,
                mask_y_cam2=mask_y_cam2,
            )
        else:
            self._impl = RealDvsVisualizer(
                event_frames_fn,
                width=width,
                height=height,
                mask_y_cam1=mask_y_cam1,
                mask_y_cam2=mask_y_cam2,
            )

    def render(self, *args, **kwargs) -> VizResult:
        return self._impl.render(*args, **kwargs)
