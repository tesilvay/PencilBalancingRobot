from .base          import StopCondition
from .fall          import FallCondition,       FallConditionParams,       FALL_CONDITION_PRESETS
from .stabilized    import StabilizedCondition, StabilizedParams,          STABILIZED_CONDITION_PRESETS
from .max_steps     import MaxStepsCondition,   MaxStepsConditionParams,   MAX_STEPS_CONDITION_PRESETS
from .any_condition import AnyStopCondition,    AnyStopConditionParams,    ANY_STOP_CONDITION_PRESETS
from .infinite      import InfiniteCondition,   InfiniteConditionParams,   INFINITE_CONDITION_PRESETS
from src.shared     import Spec,NullParams, NULL_PRESETS


STOP_CONDITION_REGISTRY = {
    "fall":       Spec(FallCondition,       FallConditionParams,     FALL_CONDITION_PRESETS),
    "stabilized": Spec(StabilizedCondition, StabilizedParams,        STABILIZED_CONDITION_PRESETS),
    "max_steps":  Spec(MaxStepsCondition,   MaxStepsConditionParams, MAX_STEPS_CONDITION_PRESETS),
    "any":        Spec(AnyStopCondition,    AnyStopConditionParams,  ANY_STOP_CONDITION_PRESETS,
                       registries={
                           "fall":       None,
                           "stabilized": None,
                           "max_steps":  None,
                       }),
    "infinite":   Spec(InfiniteCondition,   InfiniteConditionParams, INFINITE_CONDITION_PRESETS),
}
