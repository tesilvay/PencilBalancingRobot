from dataclasses import dataclass

from .base import StopCondition


@dataclass
class AnyStopConditionParams:
    conditions: dict


ANY_STOP_CONDITION_PRESETS = {
    "max_steps": {
        "conditions": {
            "max_steps": "max_steps:default",
        }
    },
    "early_stop": {
        "conditions": {
            "fall":       "fall:default",
            "stabilized": "stabilized:default",
            "max_steps":  "max_steps:default",
        }
    },
    "infinite": {
        "conditions": {
            "infinite": "infinite:default",
        }
    },
    "default": {
        "conditions": {
            "fall":       "fall:default",
            "stabilized": "stabilized:default",
            "max_steps":  "max_steps:default",
        }
    },
}


class AnyStopCondition(StopCondition):
    def __init__(self, params: AnyStopConditionParams):
        # conditions is a dict[str, StopCondition]; iterate over values
        self.conditions = list(params.conditions.values())

    def reset(self):
        for c in self.conditions:
            if hasattr(c, "reset"):
                c.reset()

    def should_stop(self, i, state, dt):
        return any(c.should_stop(i, state, dt) for c in self.conditions)

    def is_stabilized(self):
        return any(
            getattr(c, "is_stabilized", lambda: False)()
            for c in self.conditions
        )

    def settling_time(self):
        for c in self.conditions:
            if hasattr(c, "settling_time"):
                t = c.settling_time()
                if t is not None:
                    return t
        return None
