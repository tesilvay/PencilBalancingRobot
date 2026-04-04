import numpy as np
from src.shared import (
    State,
    CameraParams,
    CameraObservation,
    CameraPair,
    Measurement,
)


class VisionModelBase:

    def __init__(self, camera_params: CameraParams):
        raise NotImplementedError

    def get_y(self, state_true: State) -> Measurement:
        # get z (raw sensor measurement)
        # cams_to_measurement (reconstruct)
        raise NotImplementedError
    
    def cams_to_measurement(self, cams_camnorm: CameraPair) -> Measurement:

        b1, s1, b2, s2 = cams_camnorm.unpack()

        denom = b1 * b2 + 1.0
        if abs(denom) < 1e-8:
            denom = 1e-8

        px = (b1 * self.yr + b1 * b2 * self.xr) / denom
        py = (b2 * self.xr - b1 * b2 * self.yr) / denom
        ax = (s1 + b1 * s2) / denom
        ay = (s2 - b2 * s1) / denom

        ax = float(np.clip(ax, -np.pi / 2, np.pi / 2))
        ay = float(np.clip(ay, -np.pi / 2, np.pi / 2))

        y_meas = Measurement(
            px=px,
            py=py,
            ax=ax,
            ay=ay,
        )

        return y_meas
    
    def get_z(self) -> CameraPair:
        # get z (raw sensor measurement)
        raise NotImplementedError
    
    def project_state_to_z(self, state_true: State) -> CameraPair:
    
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
    
    def is_valid_y_meas(self, y_meas) -> bool:
        
        # 1. Numerical sanity: protects against NaNs, inf, model explosions
        if not np.all(np.isfinite([y_meas.px, y_meas.py, y_meas.ax, y_meas.ay])):
            return False
        
        return True

