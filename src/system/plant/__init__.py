from .balancer import BalancerPlant
from .null     import NullPlant
from src.shared import Spec, PlantParams, PLANT_PRESETS, NullParams, NULL_PRESETS


PLANT_REGISTRY = {
    "sim":  Spec(BalancerPlant, PlantParams, PLANT_PRESETS),
    "null": Spec(NullPlant, NullParams, NULL_PRESETS),
}
