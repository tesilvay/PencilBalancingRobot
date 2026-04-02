class SchedulerBase:
    """Base class for schedulers."""

    def tick(self):
        raise NotImplementedError

    def should_actuate(self):
        raise NotImplementedError

    def should_render(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
