import numpy as np
from src.shared import (
    SystemState,
    CameraParams,
    CameraObservation,
    CameraPair,
    PoseMeasurement,
)


def get_measurements(cams: CameraPair):
    b1 = cams.cam1.intercept
    s1 = cams.cam1.slope

    b2 = cams.cam2.intercept
    s2 = cams.cam2.slope
    
    return b1, s1, b2, s2


class VisionModelBase:

    def __init__(self, camera_params: CameraParams):
        self.xr = camera_params.xr
        self.yr = camera_params.yr


    def reconstruct(self, cams: CameraPair) -> PoseMeasurement:

        b1, s1, b2, s2 = get_measurements(cams)

        denom = b1 * b2 + 1.0
        if abs(denom) < 1e-8:
            denom = 1e-8

        X = (b1 * self.yr + b1 * b2 * self.xr) / denom
        Y = (b2 * self.xr - b1 * b2 * self.yr) / denom
        alpha_x = (s1 + b1 * s2) / denom
        alpha_y = (s2 - b2 * s1) / denom

        alpha_x = float(np.clip(alpha_x, -np.pi / 2, np.pi / 2))
        alpha_y = float(np.clip(alpha_y, -np.pi / 2, np.pi / 2))

        pose = PoseMeasurement(
            X=X,
            Y=Y,
            alpha_x=alpha_x,
            alpha_y=alpha_y,
        )

        return pose
        
    def project(self, state_true: SystemState) -> CameraPair:

        X = state_true.px
        Y = state_true.py
        alpha_x = state_true.ax
        alpha_y = state_true.ay

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
