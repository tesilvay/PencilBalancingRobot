from .system     import System, SystemParams, SYSTEM_PRESETS
from .controller import CONTROLLER_REGISTRY
from .estimator  import ESTIMATOR_REGISTRY
from .sensor     import SENSOR_REGISTRY
from .actuator   import ACTUATOR_REGISTRY
from .supervisor import SUPERVISOR_REGISTRY
from .plant      import PLANT_REGISTRY
from src.shared  import Spec

SYSTEM_REGISTRY = {
    "default": Spec(
        cls        = System,
        Params     = SystemParams,
        Presets    = SYSTEM_PRESETS,
        registries = {
            "plant":       PLANT_REGISTRY,
            "controllers": CONTROLLER_REGISTRY,
            "estimators":  ESTIMATOR_REGISTRY,
            "sensor":      SENSOR_REGISTRY,
            "actuator":    ACTUATOR_REGISTRY,
            "supervisor":  SUPERVISOR_REGISTRY,
        }
    )
}
