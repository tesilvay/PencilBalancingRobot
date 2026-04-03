from dataclasses import dataclass

from .base import StopCondition



class InfiniteCondition(StopCondition):
    def should_stop(self, i, state, dt):
        return False
