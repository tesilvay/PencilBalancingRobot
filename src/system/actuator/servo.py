from dataclasses import dataclass
import serial, time
from .base import Actuator

@dataclass
class ServoParams:
    mechanism:  Mechanism
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

    def apply(self, command) -> None:
        now = time.time()
        if now - self.last_send < self.period:
            return
        theta1, theta2 = self.mechanism.command_to_angles(command)
        self._send_raw(f"CMD,{theta1:.2f},{theta2:.2f}")
        self.last_send = now

    def reset(self) -> None:
        self.last_send = 0.0

    def _send_raw(self, cmd: str) -> None:
        self._serial.write((cmd.strip() + "\r\n").encode("utf-8"))
        self._serial.flush()