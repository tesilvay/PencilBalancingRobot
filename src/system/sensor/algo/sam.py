from dataclasses import dataclass, field

import numpy as np
from src.shared import CameraObservation, CameraParams, default_camera_params

from .base import DVSLineAlgorithm


@dataclass
class SamLineParams:
    cam_params = field(default_factory=default_camera_params)
    min_points:  int


SAM_PRESETS = {
    "default": {
        "min_points": 50,
    }
}


class SamLineAlgorithm(DVSLineAlgorithm):
    """OLS line fit on event coordinates. Line: x = slope * y + intercept."""

    def __init__(self, params: SamLineParams):
        cam = params.cam_params
        self.W = int(cam.DAVIS346_WIDTH)
        self.H = int(cam.DAVIS346_HEIGHT)
        self.min_points = params.min_points

    def update(self, events_np):
        xs = events_np["x"].astype(np.float32)
        ys = events_np["y"].astype(np.float32)

        if len(xs) < self.min_points:
            return None, None

        N    = len(xs)
        S_y  = np.sum(ys)
        S_yy = np.sum(ys * ys)
        S_x  = np.sum(xs)
        S_xy = np.sum(xs * ys)

        denom = N * S_yy - S_y * S_y
        if abs(denom) < 1e-6:
            return None, None

        slope     = (N * S_xy - S_y * S_x) / denom
        intercept = (S_x - slope * S_y) / N

        return CameraObservation(slope=slope, intercept=intercept)

    def reset(self):
        pass
