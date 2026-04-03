from dataclasses import dataclass

import numpy as np
from src.shared import (
    CameraObservation,
    CameraPair,
    CameraParams,
)

from .base import VisionModelBase, get_measurements
from src.system.sensor.observation_model.camera_model import CameraModel


@dataclass
class SimDVSParams:
    cam_params: CameraParams
    algo:       object
    obs_model:  object


SIM_DVS_PRESETS = {
    "hough": {
        "cam_params": "default:default",
        "algo":       "hough:default",
        "obs_model":  "none:default",
    },
    "sam": {
        "cam_params": "default:default",
        "algo":       "sam:default",
        "obs_model":  "none:default",
    },
}


class SimEventCameraInterface(VisionModelBase):

    def __init__(self, params: SimDVSParams):
        import copy
        cam = params.cam_params
        super().__init__(cam)

        self.cam1_algo = copy.deepcopy(params.algo)
        self.cam2_algo = copy.deepcopy(params.algo)
        dvs_mask_line_y_cam1 = int(cam.y_mask_line_1)
        dvs_mask_line_y_cam2 = int(cam.y_mask_line_2)
        self.sigma_px = 1.0
        
        self.cam = CameraModel()
        self._surface1 = np.zeros((self.cam.height, self.cam.width), dtype=np.float32)
        self._surface2 = np.zeros((self.cam.height, self.cam.width), dtype=np.float32)
        self._decay_display = 0.5
        
        self._dvs_mask_line_y_cam1 = int(dvs_mask_line_y_cam1)
        self._dvs_mask_line_y_cam2 = int(dvs_mask_line_y_cam2)
        
        # tuneable noise parameters
        self.event_density_base = 300
        self.visibility_threshold = 0.0001
        self.visibility_sharpness = 5.0

        self.pixel_noise_pct = 0.7e-2
        self.line_dropout_pct = 5e-2
        self.background_noise_pct = 70e-2

        self.center_bias_strength = 90e-2
        
    def generate_events(self, b, s, state_true, cam_id):

        mask_line_y = self._get_mask_line_y(cam_id)

        s_px, b_px = self._project_to_pixel(b, s)

        t_vals = self._sample_pencil_points()

        v_local = self._calculate_local_velocity(state_true, t_vals)

        t_vals = self._apply_visibility_model(t_vals, v_local)

        if len(t_vals) == 0:
            return self._empty_events()

        xs, ys = self._map_to_pixel_line(t_vals, s_px, b_px, mask_line_y)

        xs, ys = self._apply_line_dropout(xs, ys)

        xs, ys = self._apply_pixel_noise(xs, ys)

        xs, ys = self._add_background_noise(xs, ys)

        xs, ys = self._clip_to_image(xs, ys)

        xs, ys = self._apply_sensor_mask(xs, ys, mask_line_y)

        return self._pack_events(xs, ys)

    def _get_mask_line_y(self, cam_id):
        return self._dvs_mask_line_y_cam1 if cam_id == 1 else self._dvs_mask_line_y_cam2


    def _project_to_pixel(self, b, s):
        obs_px = self.cam.camnorm_to_pixel(CameraObservation(slope=s, intercept=b))
        return obs_px.slope, obs_px.intercept

    def _sample_pencil_points(self):
        n = self.event_density_base

        if self.center_bias_strength > 0:
            return np.random.beta(
                1 + self.center_bias_strength * 5,
                1 + self.center_bias_strength * 5,
                n
            )
        return np.random.uniform(0, 1, n)

    def _calculate_local_velocity(self, state_true, t_vals):

        xdot = state_true.x_dot
        ydot = state_true.y_dot
        ax_dot = state_true.alpha_x_dot
        ay_dot = state_true.alpha_y_dot

        return np.sqrt(
            (xdot + t_vals * ax_dot)**2 +
            (ydot + t_vals * ay_dot)**2
        )

    def _apply_visibility_model(self, t_vals, v_local):

        v0 = self.visibility_threshold
        gamma = self.visibility_sharpness

        p_event = (v_local / (v_local + v0)) ** gamma

        keep = np.random.rand(len(t_vals)) < p_event
        return t_vals[keep]

    def _map_to_pixel_line(self, t_vals, s_px, b_px, mask_line_y):
        ys = t_vals * mask_line_y
        xs = s_px * ys + b_px
        return xs, ys

    def _apply_line_dropout(self, xs, ys):
        keep = np.random.rand(len(xs)) > self.line_dropout_pct
        return xs[keep], ys[keep]


    def _apply_pixel_noise(self, xs, ys):

        cam_height = self.cam.height
        cam_width = self.cam.width

        sigma_px = self.pixel_noise_pct * max(cam_width, cam_height)

        xs = xs + np.random.normal(0, sigma_px, len(xs))
        ys = ys + np.random.normal(0, sigma_px, len(ys))

        return xs, ys

    def _add_background_noise(self, xs, ys):

        cam_height = self.cam.height
        cam_width = self.cam.width

        n_bg = int(self.background_noise_pct * len(xs))

        if n_bg == 0:
            return xs, ys

        x_bg = np.random.uniform(0, cam_width, n_bg)
        y_bg = np.random.uniform(0, cam_height, n_bg)

        xs = np.concatenate([xs, x_bg])
        ys = np.concatenate([ys, y_bg])

        return xs, ys

    def _clip_to_image(self, xs, ys):

        cam_height = self.cam.height
        cam_width = self.cam.width

        mask = (
            (xs >= 0) & (xs < cam_width) &
            (ys >= 0) & (ys < cam_height)
        )

        return xs[mask], ys[mask]

    def _apply_sensor_mask(self, xs, ys, mask_line_y):
        keep = ys < mask_line_y
        return xs[keep], ys[keep]

    def _pack_events(self, xs, ys):

        xs = xs.astype(np.int16)
        ys = ys.astype(np.int16)

        events = np.zeros(len(xs), dtype=[("x", np.int16), ("y", np.int16)])
        events["x"] = xs
        events["y"] = ys

        return events


    def _empty_events(self):
        return np.zeros(0, dtype=[("x", np.int16), ("y", np.int16)])

    def get_observation(self, state_true):

        # compute true line
        cams = super().project(state_true)

        b1, s1, b2, s2 = get_measurements(cams)

        events1 = self.generate_events(b1, s1, state_true, cam_id=1)
        events2 = self.generate_events(b2, s2, state_true, cam_id=2)
        self._surface1 *= self._decay_display
        self._surface2 *= self._decay_display
        if len(events1) > 0:
            np.add.at(self._surface1, (events1["y"], events1["x"]), 1.0)
        if len(events2) > 0:
            np.add.at(self._surface2, (events2["y"], events2["x"]), 1.0)

        result1 = self.cam1_algo.update(events1)
        result2 = self.cam2_algo.update(events2)

        # tracker not ready yet (returns (None, None) or CameraObservation)
        if isinstance(result1, tuple) or isinstance(result2, tuple):
            return None

        obs1 = self.cam.pixel_to_camnorm(result1)
        obs2 = self.cam.pixel_to_camnorm(result2)

        return CameraPair(
            CameraObservation(slope=obs1.slope, intercept=obs1.intercept),
            CameraObservation(slope=obs2.slope, intercept=obs2.intercept)
        )

    def get_surfaces(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return copy of current simulated event surfaces for visualization."""
        return self._surface1.copy(), self._surface2.copy()

    def get_event_accumulator_frames(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Alias for :meth:`get_surfaces` for visualizer compatibility."""
        return self.get_surfaces()
        
    def reset(self):
        self.cam1_algo.reset()
        self.cam2_algo.reset()
        self._surface1.fill(0.0)
        self._surface2.fill(0.0)
