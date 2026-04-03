from src.shared import Spec

from .scheduler import Scheduler, SchedulerParams, SCHEDULER_PRESETS

SCHEDULER_REGISTRY = {
    "realtime": Spec(Scheduler, SchedulerParams, SCHEDULER_PRESETS),
}
