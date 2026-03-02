import numpy as np
from collections import deque
from sim_types import (
    SystemState,
    CameraParams,
    CameraObservation,
    CameraPair,
    PoseMeasurement
)


class VisionSystem:

    def __init__(self, camera_params, noise_std=None, delay_steps=0):
        self.xr = camera_params.xr
        self.yr = camera_params.yr
        self.noise_std = noise_std
        self.delay_steps = delay_steps
        self.buffer = deque(maxlen=delay_steps + 1)

    # -------------------------------------------------
    # Project true 3D state into both camera views
    # -------------------------------------------------
    def project(self, state_true: SystemState) -> CameraPair:

        X = state_true.x
        Y = state_true.y
        alpha_x = state_true.alpha_x
        alpha_y = state_true.alpha_y

        # ----- Camera 1 (shifted along Y) -----
        # b1 = X / (Y + yr)
        # s1 = alpha_x - (X * alpha_y) / (Y + yr)

        denom1 = Y + self.yr
        if abs(denom1) < 1e-8:
            denom1 = 1e-8

        b1 = X / denom1
        s1 = alpha_x - (X * alpha_y) / denom1

        # ----- Camera 2 (shifted along X) -----
        # b2 = Y / (xr - X)
        # s2 = alpha_y + (Y * alpha_x) / (xr - X)

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

        cam1 = CameraObservation(
            slope=s1,
            intercept=b1
        )
        
        cam2 = CameraObservation(
            slope=s2,
            intercept=b2
        )
        
        cam_pair = CameraPair(cam1=cam1, cam2=cam2)

        # ---- Apply measurement delay ----
        if self.delay_steps > 0:
            self.buffer.append(cam_pair)

            # If buffer not filled yet, return current (startup phase)
            if len(self.buffer) <= self.delay_steps:
                return cam_pair

            # Otherwise return oldest buffered measurement
            return self.buffer[0]

        return cam_pair

    # -------------------------------------------------
    # Reconstruct 3D pose from two camera observations
    # -------------------------------------------------
    def reconstruct(self, cams: CameraPair) -> PoseMeasurement:

        b1 = cams.cam1.intercept
        s1 = cams.cam1.slope

        b2 = cams.cam2.intercept
        s2 = cams.cam2.slope

        denom = b1 * b2 + 1.0
        if abs(denom) < 1e-8:
            denom = 1e-8

        # From paper:
        # X  = (b1*yr + b1*b2*xr) / (b1*b2 + 1)
        # Y  = (b2*xr - b1*b2*yr) / (b1*b2 + 1)
        # alpha_x = (s1 + b1*s2) / (b1*b2 + 1)
        # alpha_y = (s2 - b2*s1) / (b1*b2 + 1)

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