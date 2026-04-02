import numpy as np


class DVSLineAlgorithm:
    """Base class for DVS line estimation algorithms."""

    def update(self, events_np):
        raise NotImplementedError

    def reset(self):
        pass
