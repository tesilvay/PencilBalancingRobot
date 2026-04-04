from .base          import StopCondition
from .fall          import FallCondition,       FallConditionParams,     FALL_CONDITION_PRESETS
from .stabilized    import StabilizedCondition, StabilizedParams,        STABILIZED_CONDITION_PRESETS
from .max_steps     import MaxStepsCondition,   MaxStepsConditionParams, MAX_STEPS_CONDITION_PRESETS
from .any_condition import AnyStopCondition,    AnyStopConditionParams,  ANY_STOP_CONDITION_PRESETS
from .infinite      import InfiniteCondition
from src.shared     import Spec, NullParams, NULL_PRESETS

# Forward-declare so AnyStopCondition can reference the full registry
STOP_CONDITION_REGISTRY: dict = {}

STOP_CONDITION_REGISTRY.update({
    "fall":       Spec(FallCondition,       FallConditionParams,     FALL_CONDITION_PRESETS),
    "stabilized": Spec(StabilizedCondition, StabilizedParams,        STABILIZED_CONDITION_PRESETS),
    "max_steps":  Spec(MaxStepsCondition,   MaxStepsConditionParams, MAX_STEPS_CONDITION_PRESETS),
    "any":        Spec(
        cls=AnyStopCondition,
        Params=AnyStopConditionParams,
        Presets=ANY_STOP_CONDITION_PRESETS,
        registries={"conditions": STOP_CONDITION_REGISTRY},
    ),
    "infinite":   Spec(InfiniteCondition, NullParams, NULL_PRESETS),
})
