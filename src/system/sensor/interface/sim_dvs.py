from dataclasses import dataclass, field

import numpy as np
from src.shared import (
    State,
    Measurement,
    CameraObservation,
    CameraPair,
    CameraParams,
    default_camera_params,
)

from .base import VisionModelBase
from src.system.sensor.observation_model.camera_model import CameraModel


@dataclass
class SimDVSParams:
    algo:       object
    obs_model:  object
    cam_params: CameraParams = field(default_factory=default_camera_params)
    


SIM_DVS_PRESETS = {
    "hough": {
        "algo":       "hough:default",
        "obs_model":  "none:default",
    },
    "sam": {
        "algo":       "sam:default",
        "obs_model":  "none:default",
    },
}


class SimEventCameraInterface(VisionModelBase):

    def __init__(self, params: SimDVSParams):
        import copy
        p = params
        cam = p.cam_params
        
        self.cam_height_px = p.cam_params.DAVIS346_HEIGHT
        self.cam_width_px = p.cam_params.DAVIS346_WIDTH

        self.cam1_algo = copy.deepcopy(p.algo)
        self.cam2_algo = copy.deepcopy(p.algo)
        
        self.sigma_px = 1.0
        
        self.cam = CameraModel()
        self._surface1 = np.zeros((self.cam_height_px, self.cam_width_px), dtype=np.float32)
        self._surface2 = np.zeros((self.cam_height_px, self.cam_width_px), dtype=np.float32)

        self._decay_display = 0.5
        self.last_line_observation = None

        self._dvs_mask_line_y_cam1 = int(cam.y_mask_line_1)
        self._dvs_mask_line_y_cam2 = int(cam.y_mask_line_2)
        
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

        xdot = state_true.vx
        ydot = state_true.vy
        ax_dot = state_true.wx
        ay_dot = state_true.wy

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

    def get_surfaces(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return copy of current simulated event surfaces for visualization."""
        return self._surface1.copy(), self._surface2.copy()

    def get_event_accumulator_frames(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Alias for :meth:`get_surfaces` for visualizer compatibility."""
        return self.get_surfaces()
      
    def _empty_events(self):
        return np.zeros(0, dtype=[("x", np.int16), ("y", np.int16)])

    def get_z(self, state_true):

        # compute true line
        cams_raw = super().project_state_to_z(state_true)

        b1, s1, b2, s2 = cams_raw.unpack()

        events1 = self.generate_events(b1, s1, state_true, cam_id=1)
        events2 = self.generate_events(b2, s2, state_true, cam_id=2)
        self._surface1 *= self._decay_display
        self._surface2 *= self._decay_display
        if len(events1) > 0:
            np.add.at(self._surface1, (events1["y"], events1["x"]), 1.0)
        if len(events2) > 0:
            np.add.at(self._surface2, (events2["y"], events2["x"]), 1.0)

        # Run algo with events
        obs1 = self.cam1_algo.update(events1)
        obs2 = self.cam2_algo.update(events2)

        # tracker not ready yet (returns (None, None) or CameraObservation)
        if isinstance(obs1, tuple) or isinstance(obs2, tuple):
            return None

        return CameraPair(
            CameraObservation(slope=obs1.slope, intercept=obs1.intercept),
            CameraObservation(slope=obs2.slope, intercept=obs2.intercept)
        )
        
    def cams_to_measurement(self, cams_px):
        
        if self.dvs_regression_model is not None:
            y_meas = self.dvs_regression_model.estimate(cams_px)

            if super.is_valid_pose(y_meas):
                return y_meas
        
        else:
            obs1 = self.cam.pixel_to_camnorm(cams_px.cam1)
            obs2 = self.cam.pixel_to_camnorm(cams_px.cam2)

            cams = CameraPair(cam1=obs1, cam2=obs2)
        
            return super().cams_to_measurement(cams_camnorm=cams)

    def get_y(self, state_true: State) -> Measurement:
        
        # returns cams in pix
        cams = self.get_z(state_true)
        self.last_line_observation = cams

        y_meas = self.cams_to_measurement(cams_px=cams)
        
        return y_meas

  
    def reset(self):
        self.last_line_observation = None
        self.cam1_algo.reset()
        self.cam2_algo.reset()
        self._surface1.fill(0.0)
        self._surface2.fill(0.0)
