from .base import GainScheduler
from .deadzone import DeadzoneGainScheduleParams, DeadzoneGainScheduler, DEADZONE_GAIN_SCHEDULE_PRESETS
from .linear_cubic import (
    LINEAR_CUBIC_GAIN_SCHEDULE_PRESETS,
    LinearCubicGainScheduleParams,
    LinearCubicGainScheduler,
)
from .null import NullGainScheduler
from .power import PowerGainScheduleParams, PowerGainScheduler, POWER_GAIN_SCHEDULE_PRESETS
from .tanh import TanhGainScheduleParams, TanhGainScheduler, TANH_GAIN_SCHEDULE_PRESETS
from src.shared import NullParams, NULL_PRESETS, Spec


GAIN_SCHEDULE_REGISTRY = {
    "deadzone": Spec(DeadzoneGainScheduler, DeadzoneGainScheduleParams, DEADZONE_GAIN_SCHEDULE_PRESETS),
    "linear_cubic": Spec(LinearCubicGainScheduler, LinearCubicGainScheduleParams, LINEAR_CUBIC_GAIN_SCHEDULE_PRESETS),
    "null": Spec(NullGainScheduler, NullParams, NULL_PRESETS),
    "tanh": Spec(TanhGainScheduler, TanhGainScheduleParams, TANH_GAIN_SCHEDULE_PRESETS),
    "power": Spec(PowerGainScheduler, PowerGainScheduleParams, POWER_GAIN_SCHEDULE_PRESETS),
}
