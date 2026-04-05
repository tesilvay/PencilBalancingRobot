from __future__ import annotations
from dataclasses import dataclass
import serial, time

import numpy as np

from .base import Actuator

@dataclass
class ServoParams:
    mechanism:  object   # Mechanism — imported at runtime to avoid circular deps
    port:       str   = "/dev/ttyUSB1"
    baud:       int   = 115200
    frequency:  float = 250.0

SERVO_PRESETS = {
    "default": {
        "mechanism": "five_bar:default",
        "port":      "/dev/ttyUSB1",
        "baud":      115200,
        "frequency": 250.0,
    }
}

class ServoActuator(Actuator):
    def __init__(self, params: ServoParams):
        self.mechanism  = params.mechanism
        self.period     = 1.0 / params.frequency
        self.last_send  = 0.0

        self._serial = serial.Serial(params.port, params.baud, timeout=1)
        time.sleep(2)
        self._send_raw("MODE,EXP")

    def mech_joint_snapshot(self, command) -> np.ndarray:
        try:
            joints, _ = self.mechanism.command_geometry(command)
            return joints
        except (ValueError, TypeError):
            return np.full((3, 2), np.nan, dtype=float)

    def apply(self, command) -> np.ndarray:
        try:
            joints, angles_deg = self.mechanism.command_geometry(command)
        except (ValueError, TypeError):
            return np.full((3, 2), np.nan, dtype=float)
        now = time.time()
        if now - self.last_send >= self.period:
            self._send_raw(f"CMD,{angles_deg[0]:.2f},{angles_deg[1]:.2f}")
            self.last_send = now
        return joints

    def reset(self) -> None:
        self.last_send = 0.0

    def _send_raw(self, cmd: str) -> None:
        self._serial.write((cmd.strip() + "\r\n").encode("utf-8"))
        self._serial.flush()