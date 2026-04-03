from .logger import Logger, TerminalInfo, SimulationResult
from src.shared import Spec, NullParams, NULL_PRESETS


LOGGER_REGISTRY = {
    "default": Spec(Logger, NullParams, NULL_PRESETS),
}
