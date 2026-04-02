from src.shared import TableCommand


class NullController:
    def compute(self, state):
        return TableCommand(0.0, 0.0)

    def reset(self):
        pass
