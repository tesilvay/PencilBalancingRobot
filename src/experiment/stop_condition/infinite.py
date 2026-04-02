from dataclasses import dataclass

from .base import StopCondition


@dataclass
class InfiniteConditionParams:
    pass


INFINITE_CONDITION_PRESETS = {"default": {}}


class InfiniteCondition(StopCondition):
    def should_stop(self, i, state, dt):
        return False
