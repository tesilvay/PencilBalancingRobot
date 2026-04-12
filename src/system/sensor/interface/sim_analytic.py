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
    bias_ang: float
    bias_pos: float
    noise_ang: float | None
    noise_pos: float | None
    cam_params: CameraParams = field(default_factory=default_camera_params)
    


SIM_ANALYTIC_PRESETS = {
    "default": {
        "noise_std":   None,
        "noise_ang":   0,
        "noise_pos":   None,
        "delay_steps": 0,
        # Add small bias to model how the real cam might add bias
        "bias_ang": 0,
        "bias_pos": 0,
        
    },
    "noisy": {
        "base": "default",
        "delay_steps": 0,
        
        "bias_ang": 0.0,
        "bias_pos": 0,
        "noise_ang": 0.1,
        "noise_pos": 0.3e-3,
        
        
    },
    "real": {
        "base": "default",
        "delay_steps": 0,
        
        "bias_ang": 0.1,
        "noise_ang": 0.2,
        
        "bias_pos": 1.5,
        "noise_pos": 2.0e-3,
    },
}


class SimVisionModel(VisionModelBase):

    def __init__(self, params: SimAnalyticParams):
        super().__init__(params.cam_params)
        self.xr = params.cam_params.xr
        self.yr = params.cam_params.yr
        
        self.bias_ang = np.deg2rad(params.bias_ang)
        self.bias_pos = params.bias_pos

        self.noise_std   = params.noise_std
        self.noise_ang = np.deg2rad(params.noise_ang)
        self.noise_pos = params.noise_pos
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
        #cams_noisy = self._add_noise(cams_raw)
        #cams = self._add_delay(cams_noisy)
        
        # turns camnorm cams into a y_meas with the analytic equations
        self.last_line_observation = cams_raw
        y_meas_raw = self.cams_to_measurement(cams_camnorm=cams_raw)
        
        y_meas = self._add_noise_y(y_meas_raw)
        
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
    
    def _add_noise_y(self, y_raw:Measurement) -> Measurement:
        
        if self.noise_pos is not None:
            y_raw.px+=np.random.normal(0, self.noise_pos) + self.bias_pos
            y_raw.py+=np.random.normal(0, self.noise_pos)
            y_raw.ax+=np.random.normal(0, self.noise_ang) + self.bias_ang
            y_raw.ay+=np.random.normal(0, self.noise_ang)
            
        
        return y_raw
    
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
