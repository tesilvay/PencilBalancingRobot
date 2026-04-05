from src.shared import NullParams, ControlInput

from .base import BaseController


class NullController(BaseController):
    def __init__(self, params: NullParams):
        pass

    def compute(self, state):
        return ControlInput(0.0, 0.0)

    def reset(self):
        pass
