from dataclasses import dataclass
from collections import deque

import numpy as np
from core.sim_types import (
    SystemState,
    CameraObservation,
    CameraPair,
)

from .base import VisionModelBase, get_measurements


@dataclass
class SimAnalyticParams:
    noise_std:   float | None
    delay_steps: int


SIM_ANALYTIC_PRESETS = {
    "default": {
        "noise_std":   None,
        "delay_steps": 0,
    },
    "noisy": {
        "base": "default",
        "noise_std":   1e-3,
        "delay_steps": 2,
    },
}


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
