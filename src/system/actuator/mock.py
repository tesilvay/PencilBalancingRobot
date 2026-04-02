from dataclasses import dataclass
from .base import Actuator

@dataclass
class MockServoParams:
    mechanism: Mechanism   # still needs mech — mock computes angles, just doesn't send

MOCK_SERVO_PRESETS = {
    "default": {
        "mechanism": "five_bar:default",
    }
}

class MockServoActuator(Actuator):
    def __init__(self, params: MockServoParams):
        self.mechanism = params.mechanism

    def apply(self, command) -> None:
        pass   # computes nothing, sends nothing — swap in real servo and it just works

    def reset(self) -> None:
        pass