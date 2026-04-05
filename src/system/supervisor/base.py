class Supervisor:
    """Base class for supervisors."""

    def update(self, x_est, innovation, dt) -> tuple[int, int]:
        """Return (controller_index, estimator_index) into System's ordered lists."""
        raise NotImplementedError

    def reset(self):
        pass
