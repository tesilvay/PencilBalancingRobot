from .base  import Actuator
from .servo import ServoController,     ServoParams,     SERVO_PRESETS
from .mock  import MockServoController, MockServoParams, MOCK_SERVO_PRESETS
from src.shared import Spec

ACTUATOR_REGISTRY = {
    "servo": Spec(ServoController,     ServoParams,     SERVO_PRESETS,      sim_only=False),
    "mock":  Spec(MockServoController, MockServoParams, MOCK_SERVO_PRESETS, sim_only=True),
}
