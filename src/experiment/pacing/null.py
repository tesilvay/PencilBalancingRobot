from .base import Pacing


class NoPacing(Pacing):
    def pace(self):
        pass
