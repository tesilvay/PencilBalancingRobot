import cv2
import numpy as np
from perception.vision import get_measurements
from perception.camera_model import CameraModel
from core.sim_types import CameraObservation, TableCommand, WorkspaceParams


class PencilVisualizerRealtime:

    def __init__(self, width=346, height=260):

        self.width = width
        self.height = height

        self.cam = CameraModel(width, height)
        
        self.cam_x = "Camera x-axis"
        self.cam_y = "Camera y-axis"
        

        cv2.namedWindow(self.cam_x, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.cam_y, cv2.WINDOW_NORMAL)

        cv2.moveWindow(self.cam_x, 50, 100)
        cv2.moveWindow(self.cam_y, 50 + self.width + 55, 137)

    def draw_line(self, img, b, s):

        # Convert normalized → pixel (line model x = s*y + b)
        obs_px = self.cam.normalized_to_pixel(CameraObservation(slope=s, intercept=b))
        s_px, b_px = obs_px.slope, obs_px.intercept

        y0 = 0
        y1 = self.height - 1

        x0 = int(s_px * y0 + b_px)
        x1 = int(s_px * y1 + b_px)

        cv2.line(img, (x0, y0), (x1, y1), 255, 2)

    def render(self, measurement, command=None, surfaces=None):
        if measurement is None:
            key = cv2.waitKey(1) & 0xFF
            return key == ord("q")

        img1 = np.zeros((self.height, self.width), dtype=np.uint8)
        img2 = np.zeros((self.height, self.width), dtype=np.uint8)

        b1, s1, b2, s2 = get_measurements(measurement)

        self.draw_line(img1, b1, s1)
        self.draw_line(img2, b2, s2)

        cv2.imshow(self.cam_x, img1)
        cv2.imshow(self.cam_y, img2)

        key = cv2.waitKey(1) & 0xFF
        return key == ord("q")


class DVSWorkspaceVisualizer:
    """
    Real DVS footage + workspace plot. Used only when real DVS cams are connected.
    Layout: Cam 1 | Cam 2 | Workspace (left to right).
    """

    def __init__(self, workspace: WorkspaceParams, width=346, height=260):
        self.width = width
        self.height = height
        self.workspace = workspace
        self.cam = CameraModel(width, height)

        self.cam_x = "Cam 1 (x-axis)"
        self.cam_y = "Cam 2 (y-axis)"
        self.workspace_win = "Workspace"

        cv2.namedWindow(self.cam_x, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.cam_y, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.workspace_win, cv2.WINDOW_NORMAL)

        cv2.moveWindow(self.cam_x, 50, 100)
        cv2.moveWindow(self.cam_y, 50 + self.width + 55, 100)
        cv2.moveWindow(self.workspace_win, 50 + 2 * self.width + 110, 100)

        self._workspace_size = 350
        self._scale = 4000  # px/m
        self._center = self._workspace_size // 2

    def _draw_line(self, frame, b, s):
        obs_px = self.cam.normalized_to_pixel(CameraObservation(slope=s, intercept=b))
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

    def _render_workspace(self, command: TableCommand | None):
        canvas = np.zeros((self._workspace_size, self._workspace_size), dtype=np.uint8)
        canvas[:] = 40
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        x_ref = self.workspace.x_ref
        y_ref = self.workspace.y_ref
        safe_radius = self.workspace.safe_radius

        if safe_radius is not None:
            radius_px = int(safe_radius * self._scale)
            cv2.circle(canvas, (self._center, self._center), radius_px, (100, 100, 100), 1)

        if command is not None:
            x_des, y_des = self._clamp_to_workspace(command.x_des, command.y_des)
            px = int(self._center + (x_des - x_ref) * self._scale)
            py = int(self._center - (y_des - y_ref) * self._scale)
            if 0 <= px < self._workspace_size and 0 <= py < self._workspace_size:
                cv2.circle(canvas, (px, py), 5, (0, 255, 0), -1)

        cv2.imshow(self.workspace_win, canvas)

    def render(self, measurement=None, command=None, surfaces=None):
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

        cv2.imshow(self.cam_x, frame1)
        cv2.imshow(self.cam_y, frame2)
        self._render_workspace(command)
        key = cv2.waitKey(1) & 0xFF
        return key == ord("q")