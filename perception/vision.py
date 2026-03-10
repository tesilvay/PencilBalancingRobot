import numpy as np
from collections import deque
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


# ============================================================
# Real DVS Camera Interface
# ============================================================

class RealEventCameraInterface(VisionModelBase):

    def __init__(self, camera_params, cam1_estimator, cam2_estimator):
        super().__init__(camera_params)

        # estimators are pluggable algorithms
        self.cam1_estimator = cam1_estimator
        self.cam2_estimator = cam2_estimator

    # -------------------------------------------------
    # Algorithm for extracting line from events
    # -------------------------------------------------
    def process_events(self, estimator, events):
        events_np = events.numpy()
        s, b = estimator.update(events_np)
        return s, b

    # -------------------------------------------------
    # Get camera pair observation from events
    # -------------------------------------------------
    def get_observation(self, events_cam1, events_cam2):

        s1, b1 = self.process_events(self.cam1_estimator, events_cam1)
        s2, b2 = self.process_events(self.cam2_estimator, events_cam2)

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

        if self.noise_std is not None:
            s1 += np.random.normal(0, self.noise_std)
            b1 += np.random.normal(0, self.noise_std)
            s2 += np.random.normal(0, self.noise_std)
            b2 += np.random.normal(0, self.noise_std)

        cam1 = CameraObservation(slope=s1, intercept=b1)
        cam2 = CameraObservation(slope=s2, intercept=b2)

        cam_pair = CameraPair(cam1=cam1, cam2=cam2)

        if self.delay_steps > 0:
            self.buffer.append(cam_pair)

            if len(self.buffer) <= self.delay_steps:
                return cam_pair

            return self.buffer[0]

        return cam_pair

    def reset(self):
        self.buffer.clear()