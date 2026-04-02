from .base    import Progress
from .console import ConsoleProgress, ProgressParams, PROGRESS_PRESETS
from src.shared import Spec

PROGRESS_REGISTRY = {
    "default": Spec(ConsoleProgress, ProgressParams, PROGRESS_PRESETS),
}
