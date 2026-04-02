from .logger import Logger
from src.shared import Spec
from src.shared import NullParams, NULL_PRESETS


LOGGER_REGISTRY = {
    "default": Spec(Logger, NullParams, NULL_PRESETS),
}
