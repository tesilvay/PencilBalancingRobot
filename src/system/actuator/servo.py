from dataclasses import dataclass
import serial
import time


@dataclass
class ServoParams:
    port:      str
    frequency: int


SERVO_PRESETS = {
    "default": {
        "port":      "/dev/ttyUSB1",
        "frequency": 250,
    }
}


class ServoController:
    """
    Low-level serial interface to Arduino.
    """

    def __init__(self, port="/dev/ttyUSB1", baud=115200):
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)

        self.send("MODE,EXP")

    def send(self, cmd):
        msg = cmd.strip() + "\r\n"
        self.ser.write(msg.encode("utf-8"))
        self.ser.flush()

    def send_angles(self, a, b):
        cmd_str = f"CMD,{a:.2f},{b:.2f}"
        self.send(cmd_str)
