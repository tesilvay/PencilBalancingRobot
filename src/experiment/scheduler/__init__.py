from .scheduler import Scheduler
from src.shared import Spec

from new_architecture.params import SchedulerParams
from new_architecture.presets import SCHEDULER_PRESETS

SCHEDULER_REGISTRY = {
    "realtime": Spec(Scheduler, SchedulerParams, SCHEDULER_PRESETS),
}
