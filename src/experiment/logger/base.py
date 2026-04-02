class LoggerBase:
    """Base class for loggers."""

    def reset(self, initial_state, initial_command):
        raise NotImplementedError

    def record(self, state, command, acc, state_est_err):
        raise NotImplementedError

    def get_result(self):
        raise NotImplementedError
