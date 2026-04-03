from src.shared import NullParams, TableCommand


class NullController:
    def __init__(self, params: NullParams):
        pass

    def compute(self, state):
        return TableCommand(0.0, 0.0)

    def reset(self):
        pass
