from .base    import Progress
from .console import ConsoleProgress, ConsoleProgressParams, ProgressParams, PROGRESS_PRESETS
from src.shared import Spec

PROGRESS_REGISTRY = {
    "default": Spec(ConsoleProgress, ConsoleProgressParams, PROGRESS_PRESETS),
}
