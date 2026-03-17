import cv2
import numpy as np
from perception.vision import get_measurements
from perception.camera_model import CameraModel
from core.sim_types import CameraObservation, TableCommand, WorkspaceParams
from visualization.composite_layout import build_composite, get_default_window_size


class PencilVisualizerRealtime:
    """
    Simulated camera view with optional workspace. Single composite window (banner + 2 or 3 panels).
    """

    _window_name = "Pencil Balancer"

    def __init__(self, width=346, height=260, show_workspace: bool = False, workspace: WorkspaceParams | None = None):
        self.width = width
        self.height = height
        self.show_workspace = show_workspace and workspace is not None
        self.workspace = workspace

        self.cam = CameraModel(width, height)

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        w, h = get_default_window_size(has_cams=True, has_workspace=self.show_workspace)
        cv2.resizeWindow(self._window_name, w, h)

        self._workspace_size = 350
        self._center = self._workspace_size // 2
        self._workspace_margin = 20
        self._grid_step_m = 0.02
        if self.show_workspace and self.workspace is not None and self.workspace.safe_radius is not None:
            self._scale = (self._workspace_size - 2 * self._workspace_margin) / (2 * self.workspace.safe_radius)
        else:
            self._scale = 4000

    def draw_line(self, img, b, s):

        # Convert normalized → pixel (line model x = s*y + b)
        obs_px = self.cam.camnorm_to_pixel(CameraObservation(slope=s, intercept=b))
        s_px, b_px = obs_px.slope, obs_px.intercept

        y0 = 0
        y1 = self.height - 1

        x0 = int(s_px * y0 + b_px)
        x1 = int(s_px * y1 + b_px)

        cv2.line(img, (x0, y0), (x1, y1), 255, 2)

    def _clamp_to_workspace(self, x_des: float, y_des: float) -> tuple[float, float]:
        if self.workspace is None:
            return x_des, y_des
        x_ref = self.workspace.x_ref
        y_ref = self.workspace.y_ref
        safe_radius = self.workspace.safe_radius
        if safe_radius is None:
            return x_des, y_des
        dx = x_des - x_ref
        dy = y_des - y_ref
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > safe_radius and dist > 0:
            scale = safe_radius / dist
            dx *= scale
            dy *= scale
            x_des = x_ref + dx
            y_des = y_ref + dy
        return x_des, y_des

    def _build_workspace_canvas(self, command: TableCommand | None) -> np.ndarray | None:
        if self.workspace is None:
            return None
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
        cv2.line(canvas, (self._center - cross_len, self._center), (self._center + cross_len, self._center), circle_color, 1)
        cv2.line(canvas, (self._center, self._center - cross_len), (self._center, self._center + cross_len), circle_color, 1)

        if command is not None:
            x_des, y_des = self._clamp_to_workspace(command.x_des, command.y_des)
            px = int(self._center + (x_des - x_ref) * self._scale)
            py = int(self._center - (y_des - y_ref) * self._scale)
            if 0 <= px < self._workspace_size and 0 <= py < self._workspace_size:
                cv2.circle(canvas, (px, py), 5, (0, 255, 0), -1)
        return canvas

    def render(self, measurement, command=None, surfaces=None, title: str | None = None, **kwargs):
        if measurement is None:
            key = cv2.waitKey(1) & 0xFF
            return key == ord("q")

        img1 = np.zeros((self.height, self.width), dtype=np.uint8)
        img2 = np.zeros((self.height, self.width), dtype=np.uint8)

        b1, s1, b2, s2 = get_measurements(measurement)

        self.draw_line(img1, b1, s1)
        self.draw_line(img2, b2, s2)

        frame1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        frame2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        workspace_canvas = self._build_workspace_canvas(command) if self.show_workspace else None
        title_str = title if title is not None else "Experiment | Q: quit"
        composite = build_composite(title_str, frame1, frame2, workspace_canvas)
        cv2.imshow(self._window_name, composite)

        key = cv2.waitKey(1) & 0xFF
        return key == ord("q")


class DVSWorkspaceVisualizer:
    """
    Real DVS footage + optional workspace plot. Single composite window (banner + 2 or 3 panels).
    """

    _window_name = "Pencil Balancer"

    def __init__(self, workspace: WorkspaceParams, width=346, height=260, show_workspace: bool = True):
        self.width = width
        self.height = height
        self.workspace = workspace
        self.show_workspace = show_workspace
        self.cam = CameraModel(width, height)

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        w, h = get_default_window_size(has_cams=True, has_workspace=show_workspace)
        cv2.resizeWindow(self._window_name, w, h)

        self._workspace_size = 350
        self._center = self._workspace_size // 2
        self._workspace_margin = 20
        self._grid_step_m = 0.02
        if workspace.safe_radius is not None:
            self._scale = (self._workspace_size - 2 * self._workspace_margin) / (2 * workspace.safe_radius)
        else:
            self._scale = 4000

    def _draw_line(self, frame, b, s):
        obs_px = self.cam.camnorm_to_pixel(CameraObservation(slope=s, intercept=b))
        s_px, b_px = obs_px.slope, obs_px.intercept
        y0, y1 = 0, self.height - 1
        x0 = int(s_px * y0 + b_px)
        x1 = int(s_px * y1 + b_px)
        cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

    def _clamp_to_workspace(self, x_des: float, y_des: float) -> tuple[float, float]:
        """Project (x_des, y_des) onto workspace circle edge if outside (same as plant.clamp_command)."""
        x_ref = self.workspace.x_ref
        y_ref = self.workspace.y_ref
        safe_radius = self.workspace.safe_radius
        if safe_radius is None:
            return x_des, y_des
        dx = x_des - x_ref
        dy = y_des - y_ref
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > safe_radius and dist > 0:
            scale = safe_radius / dist
            dx *= scale
            dy *= scale
            x_des = x_ref + dx
            y_des = y_ref + dy
        return x_des, y_des

    def _build_workspace_canvas(self, command: TableCommand | None, paused: bool = False) -> np.ndarray:
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
        cv2.line(canvas, (self._center - cross_len, self._center), (self._center + cross_len, self._center), circle_color, 1)
        cv2.line(canvas, (self._center, self._center - cross_len), (self._center, self._center + cross_len), circle_color, 1)

        if command is not None:
            x_des, y_des = self._clamp_to_workspace(command.x_des, command.y_des)
            px = int(self._center + (x_des - x_ref) * self._scale)
            py = int(self._center - (y_des - y_ref) * self._scale)
            if 0 <= px < self._workspace_size and 0 <= py < self._workspace_size:
                cv2.circle(canvas, (px, py), 5, (0, 255, 0), -1)

        return canvas

    def render(self, measurement=None, command=None, surfaces=None, title: str | None = None, paused: bool | None = None):
        if surfaces is not None and len(surfaces) == 2:
            surface1, surface2 = surfaces
            frame1 = np.clip(surface1 * 50, 0, 255).astype(np.uint8)
            frame2 = np.clip(surface2 * 50, 0, 255).astype(np.uint8)
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)
        else:
            frame1 = np.zeros((self.height, self.width), dtype=np.uint8)
            frame2 = np.zeros((self.height, self.width), dtype=np.uint8)
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)

        if measurement is not None:
            b1, s1, b2, s2 = get_measurements(measurement)
            self._draw_line(frame1, b1, s1)
            self._draw_line(frame2, b2, s2)

        is_paused = paused is True
        workspace_canvas = self._build_workspace_canvas(command, paused=is_paused) if self.show_workspace else None
        title_str = title if title is not None else (
            "Paused - table at center | Space: resume | Q: quit" if is_paused else "Experiment | Space: pause | Q: quit"
        )
        composite = build_composite(title_str, frame1, frame2, workspace_canvas)
        cv2.imshow(self._window_name, composite)

        key = cv2.waitKey(1) & 0xFF
        quit_requested = key == ord("q")
        toggle_pause = key == ord(" ")
        return (quit_requested, toggle_pause)