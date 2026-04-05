from src.shared import ControlInput


class BaseController:
    def compute(self, state):
        raise NotImplementedError

    def reset(self):
        """Reset controller memory between trials (smooth controllers, etc.)."""
        pass

    def set_applied_command(self, u: ControlInput) -> None:
        """Sync internal memory to the command actually used after post-processing (e.g. workspace clamp)."""
        pass
