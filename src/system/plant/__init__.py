from .balancer import BalancerPlant, BalancerParams, BALANCER_PRESETS
from .null     import NullPlant
from src.shared import Spec, NullParams, NULL_PRESETS


PLANT_REGISTRY = {
    "sim":  Spec(BalancerPlant, BalancerParams, BALANCER_PRESETS),
    "null": Spec(NullPlant,     NullParams,     NULL_PRESETS),
}
