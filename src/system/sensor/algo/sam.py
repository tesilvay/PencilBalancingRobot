from dataclasses import dataclass

import numpy as np
from src.shared import CameraObservation, CameraParams
from perception.dvs_camera_reader import DAVIS346_WIDTH, DAVIS346_HEIGHT

from .base import DVSLineAlgorithm


@dataclass
class SamLineParams:
    cam_params:       CameraParams
    min_points: int


SAM_PRESETS = {
    "default": {
        "cam_params": "default:default",
        "min_points": 50,
    }
}


class SamLineAlgorithm(DVSLineAlgorithm):
    """
    OLS line fit on event coordinates (from sam_cam.py).
    Line: x = slope * y + intercept.
    Fits directly to event batch; no surface accumulation.
    """

    def __init__(self, width=DAVIS346_WIDTH, height=DAVIS346_HEIGHT, min_points=50):
        self.W = width
        self.H = height
        self.min_points = min_points

    def update(self, events_np):
        xs = events_np["x"].astype(np.float32)
        ys = events_np["y"].astype(np.float32)

        if len(xs) < self.min_points:
            return None, None

        N = len(xs)
        S_y = np.sum(ys)
        S_yy = np.sum(ys * ys)
        S_x = np.sum(xs)
        S_xy = np.sum(xs * ys)

        denom = N * S_yy - S_y * S_y

        if abs(denom) < 1e-6:
            return None, None

        slope = (N * S_xy - S_y * S_x) / denom
        intercept = (S_x - slope * S_y) / N

        return CameraObservation(slope=slope, intercept=intercept)
