from dataclasses import dataclass

from .base import StopCondition



class InfiniteCondition(StopCondition):
    def __init__(self, params=None):
        pass

    def should_stop(self, i, state, dt):
        return False
