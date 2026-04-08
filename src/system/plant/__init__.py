from .balancer import BalancerPlant, BalancerParams, BALANCER_PRESETS
from .accel_follow import AccelFollowPlant, AccelFollowParams, ACCEL_FOLLOW_PRESETS
from .placing import PlacingPlant, PlacingParams, PLACING_PRESETS
from .null     import NullPlant
from src.shared import Spec, NullParams, NULL_PRESETS


PLANT_REGISTRY = {
    "accel_sim": Spec(AccelFollowPlant, AccelFollowParams, ACCEL_FOLLOW_PRESETS),
    "sim":     Spec(BalancerPlant, BalancerParams, BALANCER_PRESETS),
    "placing": Spec(PlacingPlant,  PlacingParams,  PLACING_PRESETS),
    "null":    Spec(NullPlant,     NullParams,     NULL_PRESETS),
}
