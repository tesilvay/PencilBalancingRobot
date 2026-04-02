from .base    import Supervisor
from .dynamic import DynamicSupervisor, DynamicSupervisorParams, DYNAMIC_SUPERVISOR_PRESETS
from .static  import StaticSupervisor,  StaticSupervisorParams,  STATIC_SUPERVISOR_PRESETS
from src.shared import Spec

SUPERVISOR_REGISTRY = {
    "dynamic": Spec(DynamicSupervisor, DynamicSupervisorParams, DYNAMIC_SUPERVISOR_PRESETS),
    "static":  Spec(StaticSupervisor,  StaticSupervisorParams,  STATIC_SUPERVISOR_PRESETS),
}
