from .experiment      import Experiment
from .logger          import LOGGER_REGISTRY
from .stop_condition  import STOP_CONDITION_REGISTRY
from .visualizer      import VISUALIZER_REGISTRY
from .progress        import PROGRESS_REGISTRY
from .pacing          import PACING_REGISTRY
from .scheduler       import SCHEDULER_REGISTRY
from src.shared       import Spec
from src.system       import SYSTEM_REGISTRY

from new_architecture.params import ExperimentParams
from new_architecture.presets import EXPERIMENT_PRESETS

EXPERIMENT_REGISTRY = {
    "default": Spec(
        cls        = Experiment,
        Params     = ExperimentParams,
        Presets    = EXPERIMENT_PRESETS,
        registries = {
            "system":         SYSTEM_REGISTRY,
            "logger":         LOGGER_REGISTRY,
            "stop_condition": STOP_CONDITION_REGISTRY,
            "visualizer":     VISUALIZER_REGISTRY,
            "progress":       PROGRESS_REGISTRY,
            "pacing":         PACING_REGISTRY,
            "scheduler":      SCHEDULER_REGISTRY,
        }
    )
}
