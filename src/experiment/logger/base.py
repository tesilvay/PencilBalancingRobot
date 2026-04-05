class LoggerBase:
    """Base class for loggers."""

    def reset(self, initial_state, initial_command):
        raise NotImplementedError

    def record(self, step_data):
        raise NotImplementedError

    def get_result(self):
        raise NotImplementedError
