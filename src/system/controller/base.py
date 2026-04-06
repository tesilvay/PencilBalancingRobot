from src.shared import ControlInput, State


class BaseController:
    def compute(self, state):
        raise NotImplementedError

    def reset(self, x_hat: State | None = None):
        """Reset controller memory between trials (smooth controllers, etc.)."""
        del x_hat
        pass

    def set_applied_command(self, u: ControlInput) -> None:
        """Sync internal memory to the command actually used after post-processing (e.g. workspace clamp)."""
        pass
