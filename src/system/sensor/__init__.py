"""
Sensor package: vision interfaces, line algorithms, and regression models.

SENSOR_REGISTRY is aliased to VISION_INTERFACE_REGISTRY so the key "sensor"
in SYSTEM_PRESETS resolves directly to an interface (e.g. "sim_dvs:hough").
"""

from .algo              import LINE_ALGO_REGISTRY
from .observation_model import REG_MODEL_REGISTRY
from .interface         import VISION_INTERFACE_REGISTRY

VISION_REGISTRY = VISION_INTERFACE_REGISTRY
SENSOR_REGISTRY = VISION_REGISTRY
