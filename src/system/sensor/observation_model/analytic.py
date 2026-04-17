from dataclasses import dataclass, field

from src.shared import CameraPair, Measurement, default_camera_params, CameraParams

from .base import RegressionModel
from src.system.sensor.observation_model.camera_model import CameraModel
import numpy as np



@dataclass
class AnalyticModelParams:
    var: float
    cam_params: CameraParams = field(default_factory=default_camera_params)



ANALYTIC_PRESETS = {
    "default": {
        "var": 30.0,
    }
}


class AnalyticModel(RegressionModel):
    """Registry-facing wrapper: loads a frozen SimpleDVSRegressionModel from disk."""

    def __init__(self, params: AnalyticModelParams):
        self.cam = CameraModel()
        self.xr = params.cam_params.xr
        self.yr = params.cam_params.yr

    def _pixel_line_to_top_origin_camnorm(self, obs_px):
        """
        Convert pixel x = s*y + b into normalized line coefficients while
        preserving the intercept at pixel y=0.

        CameraModel.pixel_to_camnorm returns the intercept at normalized y=0,
        which is pixel y=cy. The analytic real-DVS equations use the line's
        top-of-image intercept instead; simple DVS regression still consumes
        pixel lines and evaluates x at the mask line separately.
        """
        s_px = float(obs_px.slope)
        b_px = float(obs_px.intercept)
        s = s_px * self.cam.fy / self.cam.fx
        b = (b_px - self.cam.cx) / self.cam.fx
        return b, s

    def estimate(self, cams_px: CameraPair) -> Measurement:
        b1, s1 = self._pixel_line_to_top_origin_camnorm(cams_px.cam1)
        b2, s2 = self._pixel_line_to_top_origin_camnorm(cams_px.cam2)

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
            px=px/-1000,
            py=py/1000,
            ax=-ax,
            ay=ay,
        )

        return y_meas
