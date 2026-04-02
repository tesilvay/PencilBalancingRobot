from .base     import Pacing
from .realtime import RealTimePacing, RealTimePacingParams, REALTIME_PACING_PRESETS
from .null     import NoPacing
from src.shared import NullParams, NULL_PRESETS

from src.shared import Spec

PACING_REGISTRY = {
    "realtime": Spec(RealTimePacing, RealTimePacingParams, REALTIME_PACING_PRESETS),
    "null":     Spec(NoPacing,       NullParams,     NULL_PRESETS),
}
