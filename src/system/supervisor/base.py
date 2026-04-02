class Supervisor:
    """Base class for supervisors."""

    def update(self, x_est, innovation, dt) -> tuple[str, str]:
        raise NotImplementedError

    def reset(self):
        pass
