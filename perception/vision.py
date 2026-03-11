import numpy as np
from collections import deque
from perception.camera_model import CameraModel
from core.sim_types import (
    SystemState,
    CameraParams,
    CameraObservation,
    CameraPair,
    PoseMeasurement
)

def get_measurements(cams: CameraPair):
    b1 = cams.cam1.intercept
    s1 = cams.cam1.slope

    b2 = cams.cam2.intercept
    s2 = cams.cam2.slope
    
    return b1, s1, b2, s2

# ============================================================
# Base Vision Model (shared math)
# ============================================================

class VisionModelBase:

    def __init__(self, camera_params: CameraParams):
        self.xr = camera_params.xr
        self.yr = camera_params.yr

    # -------------------------------------------------
    # Reconstruct 3D pose from two camera observations
    # -------------------------------------------------
    def reconstruct(self, cams: CameraPair) -> PoseMeasurement:

        b1, s1, b2, s2 = get_measurements(cams)

        denom = b1 * b2 + 1.0
        if abs(denom) < 1e-8:
            denom = 1e-8

        # some sort of projection down to base height instead of camera height at b1, b2?
        X = (b1 * self.yr + b1 * b2 * self.xr) / denom
        Y = (b2 * self.xr - b1 * b2 * self.yr) / denom
        alpha_x = (s1 + b1 * s2) / denom
        alpha_y = (s2 - b2 * s1) / denom

        return PoseMeasurement(
            X=X,
            Y=Y,
            alpha_x=alpha_x,
            alpha_y=alpha_y
        )
        
    def project(self, state_true: SystemState) -> CameraPair:

        X = state_true.x
        Y = state_true.y
        alpha_x = state_true.alpha_x
        alpha_y = state_true.alpha_y

        denom1 = Y + self.yr
        if abs(denom1) < 1e-8:
            denom1 = 1e-8

        b1 = X / denom1
        s1 = alpha_x - (X * alpha_y) / denom1

        denom2 = self.xr - X
        if abs(denom2) < 1e-8:
            denom2 = 1e-8

        b2 = Y / denom2
        s2 = alpha_y + (Y * alpha_x) / denom2

        cam1 = CameraObservation(slope=s1, intercept=b1)
        cam2 = CameraObservation(slope=s2, intercept=b2)

        return CameraPair(cam1=cam1, cam2=cam2)


# ============================================================
# Real DVS Camera Interface
# ============================================================

class RealEventCameraInterface(VisionModelBase):

    def __init__(self, camera_params, cam1_algo, cam2_algo):
        super().__init__(camera_params)

        # algorithms are pluggable
        self.cam1_algo = cam1_algo
        self.cam2_algo = cam2_algo

    # -------------------------------------------------
    # Algorithm for extracting line from events
    # -------------------------------------------------
    def process_events(self, algo, events):
        events_np = events.numpy()
        s, b = algo.update(events_np)
        return s, b

    # -------------------------------------------------
    # Get camera pair observation from events
    # -------------------------------------------------
    def get_observation(self, events_cam1, events_cam2):

        s1, b1 = self.process_events(self.cam1_algo, events_cam1)
        s2, b2 = self.process_events(self.cam2_algo, events_cam2)

        if s1 is None or s2 is None:
            return None

        cam1 = CameraObservation(
            slope=s1,
            intercept=b1
        )

        cam2 = CameraObservation(
            slope=s2,
            intercept=b2
        )

        return CameraPair(cam1=cam1, cam2=cam2)

# ============================================================
# Simulated DVS Camera Interface
# ============================================================
class SimEventCameraInterface(VisionModelBase):

    def __init__(self, camera_params, cam1_algo, cam2_algo):
        super().__init__(camera_params)

        self.cam1_algo = cam1_algo
        self.cam2_algo = cam2_algo
        
        self.cam = CameraModel()

    def generate_events(self, b, s, n=200):

        # convert normalized line → pixel line
        a_px, b_px = self.cam.normalized_to_pixel(b, s)

        cam_height = self.cam.height
        cam_width = self.cam.width

        # sample points along the line
        ys = np.random.uniform(0, cam_height, n)
        xs = a_px * ys + b_px

        # add pixel noise
        xs += np.random.normal(0, 1.0, n)

        # keep only events inside the image
        mask = (xs >= 0) & (xs < cam_width)
        xs = xs[mask]
        ys = ys[mask]

        xs = xs.astype(np.int16)
        ys = ys.astype(np.int16)

        events = np.zeros(len(xs), dtype=[("x", np.int16), ("y", np.int16)])
        events["x"] = xs
        events["y"] = ys

        return events

    def get_observation(self, state_true):

        # compute true line
        cams = super().project(state_true)

        b1, s1, b2, s2 = get_measurements(cams)

        events1 = self.generate_events(b1, s1)
        events2 = self.generate_events(b2, s2)

        b1_est_pix, s1_est_pix = self.cam1_algo.update(events1)
        b2_est_pix, s2_est_pix = self.cam2_algo.update(events2)

        # tracker not ready yet
        if s1_est_pix is None or s2_est_pix is None:
            return None

        b1_est, s1_est = self.cam.pixel_to_normalized(b1_est_pix, s1_est_pix)
        b2_est, s2_est = self.cam.pixel_to_normalized(b2_est_pix, s2_est_pix)
        
        return CameraPair(
            CameraObservation(s1_est, b1_est),
            CameraObservation(s2_est, b2_est)
        )
        
    def reset(self):
        self.cam1_algo.reset()
        self.cam2_algo.reset()


# ============================================================
# Simulated Vision Model
# ============================================================

class SimVisionModel(VisionModelBase):

    def __init__(self, camera_params, noise_std=None, delay_steps=0):
        super().__init__(camera_params)

        self.noise_std = noise_std
        self.delay_steps = delay_steps
        self.buffer = deque(maxlen=delay_steps + 1)

    # -------------------------------------------------
    # Project true 3D state into both camera views
    # -------------------------------------------------
    def get_observation(self, state_true: SystemState) -> CameraPair:
        
        cams = super().project(state_true)
        
        noisy_cams = self._add_noise(cams)

        if self.delay_steps > 0:
            self.buffer.append(noisy_cams)

            if len(self.buffer) <= self.delay_steps:
                return noisy_cams

            return self.buffer[0]

        return noisy_cams
    
    def _add_noise(self, cams: CameraPair):
        
        b1, s1, b2, s2 = get_measurements(cams)
        
        if self.noise_std is not None:
            s1 += np.random.normal(0, self.noise_std)
            b1 += np.random.normal(0, self.noise_std)
            s2 += np.random.normal(0, self.noise_std)
            b2 += np.random.normal(0, self.noise_std)
        
        cam1 = CameraObservation(slope=s1, intercept=b1)
        cam2 = CameraObservation(slope=s2, intercept=b2)

        noisy_cams = CameraPair(cam1=cam1, cam2=cam2)
        
        return noisy_cams

    def reset(self):
        self.buffer.clear()