from core.sim_types import SystemState, TableCommand


class BaseController:
    def compute(self, state):
        raise NotImplementedError

    def reset(self):
        """Reset controller memory between trials (smooth controllers, etc.)."""
        pass
