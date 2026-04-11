class Pacing:
    """Base class for pacing strategies."""

    def reset(self):
        pass

    def pace(self):
        raise NotImplementedError
