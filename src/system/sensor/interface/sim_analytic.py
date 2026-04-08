from dataclasses import dataclass, field
from collections import deque

import numpy as np
from src.shared import (
    State,
    CameraPair,
    Measurement,
    CameraParams,
    default_camera_params,
)

from .base import VisionModelBase


@dataclass
class SimAnalyticParams:
    noise_std:   float | None
    delay_steps: int
    cam_params: CameraParams = field(default_factory=default_camera_params)


SIM_ANALYTIC_PRESETS = {
    "default": {
        "noise_std":   None,
        "delay_steps": 0,
    },
    "noisy": {
        "base": "default",
        "noise_std":   1e-5,
        "delay_steps": 0,
    },
    "noisiest": {
        "base": "default",
        "noise_std":   1e-2,
        "delay_steps": 0,
    },
}


class SimVisionModel(VisionModelBase):

    def __init__(self, params: SimAnalyticParams):
        super().__init__(params.cam_params)
        self.xr = params.cam_params.xr
        self.yr = params.cam_params.yr

        self.noise_std   = params.noise_std
        self.delay_steps = params.delay_steps
        self.buffer = deque(maxlen=params.delay_steps + 1)
        self.last_line_observation = None

    # -------------------------------------------------
    # Project true 3D state into both camera views
    # -------------------------------------------------
    
    def get_y(self, state_true: State) -> Measurement:
        
        # returns cams in camnorm
        cams_raw = self.get_z(state_true)
        
        # add noise and delay to simulate realism
        cams_noisy = self._add_noise(cams_raw)
        cams = self._add_delay(cams_noisy)
        
        # turns camnorm cams into a y_meas with the analytic equations
        self.last_line_observation = cams
        y_meas = self.cams_to_measurement(cams_camnorm=cams)
        
        return y_meas
    
    def _add_delay(self, noisy_cams: CameraPair) -> CameraPair:
        
        if self.delay_steps > 0:
            self.buffer.append(noisy_cams)

            if len(self.buffer) <= self.delay_steps:
                return noisy_cams

            return self.buffer[0]

        return noisy_cams
    
    def _add_noise(self, cams: CameraPair) -> CameraPair:
         
        if self.noise_std is not None:
            cams.cam1.slope += np.random.normal(0, self.noise_std)
            cams.cam1.intercept += np.random.normal(0, self.noise_std)
            cams.cam2.slope += np.random.normal(0, self.noise_std)
            cams.cam2.intercept += np.random.normal(0, self.noise_std)
        
        return cams
    
    def get_z(self, state_true: State) -> CameraPair:

        return super().project_state_to_z(state_true)
    
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
    

    def reset(self):
        self.last_line_observation = None
        self.buffer.clear()
