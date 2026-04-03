from src.shared import NullParams

from .base import Pacing


class NoPacing(Pacing):
    def __init__(self, params: NullParams):
        pass

    def pace(self):
        pass
