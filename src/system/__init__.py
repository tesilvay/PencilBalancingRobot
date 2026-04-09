from .system import (
    System,
    SystemParams,
    SYSTEM_PRESETS,
)
from .real_system import(
    RealSystem,
    RealSystemParams,
    REAL_SYSTEM_PRESETS,
)
from .controller import CONTROLLER_REGISTRY
from .estimator  import ESTIMATOR_REGISTRY
from .sensor     import SENSOR_REGISTRY
from .actuator   import ACTUATOR_REGISTRY
from .supervisor import SUPERVISOR_REGISTRY
from .plant      import PLANT_REGISTRY
from .gain_scheduling import GAIN_SCHEDULE_REGISTRY
from src.shared  import Spec

SYSTEM_REGISTRY = {
    "default": Spec(
        cls        = System,
        Params     = SystemParams,
        Presets    = SYSTEM_PRESETS,
        registries = {
            "plants":       PLANT_REGISTRY,
            "controllers": CONTROLLER_REGISTRY,
            "estimators":  ESTIMATOR_REGISTRY,
            "sensor":      SENSOR_REGISTRY,
            "actuator":    ACTUATOR_REGISTRY,
            "supervisor":  SUPERVISOR_REGISTRY,
            "gain_schedule": GAIN_SCHEDULE_REGISTRY,
        }
    ),
    "real": Spec(
        cls        = RealSystem,
        Params     = RealSystemParams,
        Presets    = REAL_SYSTEM_PRESETS,
        registries = {
            "plants":       PLANT_REGISTRY,
            "controllers": CONTROLLER_REGISTRY,
            "estimators":  ESTIMATOR_REGISTRY,
            "sensor":      SENSOR_REGISTRY,
            "actuator":    ACTUATOR_REGISTRY,
            "supervisor":  SUPERVISOR_REGISTRY,
            "gain_schedule": GAIN_SCHEDULE_REGISTRY,
        }
    )
}
