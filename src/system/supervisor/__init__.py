from .base    import Supervisor
from .dynamic import DynamicSupervisor, DynamicSupervisorParams, DYNAMIC_SUPERVISOR_PRESETS
from .real_dynamic import (
    RealDynamicSupervisor,
    RealDynamicSupervisorParams,
    REAL_DYNAMIC_SUPERVISOR_PRESETS,
)
from .real    import RealSupervisor, RealSupervisorParams, REAL_SUPERVISOR_PRESETS
from .static  import StaticSupervisor,  StaticSupervisorParams,  STATIC_SUPERVISOR_PRESETS
from src.shared import Spec

SUPERVISOR_REGISTRY = {
    "dynamic": Spec(DynamicSupervisor, DynamicSupervisorParams, DYNAMIC_SUPERVISOR_PRESETS),
    "real_dynamic": Spec(RealDynamicSupervisor, RealDynamicSupervisorParams, REAL_DYNAMIC_SUPERVISOR_PRESETS),
    "real":    Spec(RealSupervisor, RealSupervisorParams, REAL_SUPERVISOR_PRESETS),
    "static":  Spec(StaticSupervisor,  StaticSupervisorParams,  STATIC_SUPERVISOR_PRESETS),
}
