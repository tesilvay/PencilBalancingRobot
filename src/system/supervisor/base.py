class Supervisor:
    """Base class for supervisors."""

    def update(self, x_est, innovation, dt) -> tuple[int, int]:
        """Return (controller_index, estimator_index) into System's ordered lists."""
        raise NotImplementedError

    @property
    def last_transition(self) -> dict | None:
        """Metadata from last update, if available."""
        return None

    def reset(self):
        pass
