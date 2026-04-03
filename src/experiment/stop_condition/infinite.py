from src.shared import NullParams

from .base import StopCondition


class InfiniteCondition(StopCondition):
    def __init__(self, params: NullParams):
        pass

    def should_stop(self, i, state, dt):
        return False
