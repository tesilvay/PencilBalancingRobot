from .vision     import Vision, VisionParams, VISION_PRESETS
from .interface  import VISION_INTERFACE_REGISTRY
from .algo       import LINE_ALGO_REGISTRY
from .observation_model import REG_MODEL_REGISTRY
from src.shared  import Spec

SENSOR_REGISTRY = {
    "default": Spec(
        cls        = Vision,
        Params     = VisionParams,
        Presets    = VISION_PRESETS,
        registries = {
            "interface": VISION_INTERFACE_REGISTRY,
            "algo":      LINE_ALGO_REGISTRY,
            "reg_model": REG_MODEL_REGISTRY,
        },
    )
}
