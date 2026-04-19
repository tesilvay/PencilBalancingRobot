from .base import Actuator
from .mech import (
    Mechanism, MechanismParams, MECHANISM_PRESETS,
    MechanismTPS, MechanismTPSParams, MECHANISM_TPS_PRESETS,
    MECHANISM_REGISTRY,
)
from .servo import ServoActuator, ServoParams, SERVO_PRESETS
from .mock  import MockServoActuator, MockServoParams, MOCK_SERVO_PRESETS
from src.shared import Spec

ACTUATOR_REGISTRY = {
    "servo": Spec(ServoActuator,     ServoParams,     SERVO_PRESETS,
                  registries={"mechanism": MECHANISM_REGISTRY}, sim_only=False),
    "mock":  Spec(MockServoActuator, MockServoParams, MOCK_SERVO_PRESETS,
                  registries={"mechanism": MECHANISM_REGISTRY}, sim_only=True),
}