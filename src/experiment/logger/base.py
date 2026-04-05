class LoggerBase:
    """Base class for loggers."""

    def reset(self, initial_step_data):
        """Seed the log from the system’s post-:meth:`~src.system.system.System.reset` snapshot."""
        raise NotImplementedError

    def record(self, step_data):
        raise NotImplementedError

    def get_result(self):
        raise NotImplementedError
