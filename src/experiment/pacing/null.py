from .base import Pacing


class NoPacing(Pacing):
    def __init__(self, params=None):
        pass

    def pace(self):
        pass
