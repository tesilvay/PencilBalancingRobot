from .system      import System
from .controller  import CONTROLLER_REGISTRY
from .estimator   import ESTIMATOR_REGISTRY
from .sensor      import SENSOR_REGISTRY
from .actuator    import ACTUATOR_REGISTRY
from .supervisor  import SUPERVISOR_REGISTRY
from src.shared   import Spec

from new_architecture.params import SystemParams
from new_architecture.presets import SYSTEM_PRESETS

SYSTEM_REGISTRY = {
    "default": Spec(
        cls        = System,
        Params     = SystemParams,
        Presets    = SYSTEM_PRESETS,
        registries = {
            "controllers": CONTROLLER_REGISTRY,
            "estimators":  ESTIMATOR_REGISTRY,
            "sensor":      SENSOR_REGISTRY,
            "actuator":    ACTUATOR_REGISTRY,
            "supervisor":  SUPERVISOR_REGISTRY,
        }
    )
}
