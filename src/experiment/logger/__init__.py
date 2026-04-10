from .logger import Logger, LoggerParams, TerminalInfo, SimulationResult
from src.shared import Spec


LOGGER_REGISTRY = {
    "default": Spec(Logger, LoggerParams, {"default": {}}),
}
