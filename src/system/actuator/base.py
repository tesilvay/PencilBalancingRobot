class Actuator:
    """Base class for actuators."""

    def send(self, command):
        raise NotImplementedError

    def reset(self):
        pass
