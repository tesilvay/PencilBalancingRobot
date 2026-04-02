from dataclasses import dataclass
import time


@dataclass
class MockServoParams:
    pass


MOCK_SERVO_PRESETS = {"default": {}}


class MockServoController:

    def __init__(self):
        self.start = time.perf_counter()

    def reset(self):
        pass

    def send(self, cmd):
        pass

    def send_angles(self, theta1, theta2):
        cmd = f"CMD,{theta1:.2f},{theta2:.2f}"
        self.send(cmd)
