class Progress:
    """Base class for progress reporters."""

    def start(self, total, label=""):
        raise NotImplementedError

    def update(self, step):
        raise NotImplementedError

    def finish(self):
        raise NotImplementedError
