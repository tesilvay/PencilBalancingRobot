from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .base import Actuator

@dataclass
class MockServoParams:
    mechanism: object   # Mechanism — still needs mech for angle computation

MOCK_SERVO_PRESETS = {
    "default": {
        "mechanism": "five_bar:default",
    }
}

class MockServoActuator(Actuator):
    def __init__(self, params: MockServoParams):
        self.mechanism = params.mechanism

    def mech_joint_snapshot(self, command) -> np.ndarray:
        try:
            joints, _ = self.mechanism.command_geometry(command)
            return joints
        except (ValueError, TypeError):
            return np.full((3, 2), np.nan, dtype=float)

    def apply(self, command) -> np.ndarray:
        return self.mech_joint_snapshot(command)

    def reset(self) -> None:
        pass