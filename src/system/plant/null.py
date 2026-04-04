from src.shared import NullParams, State, TableAccel, ControlInput


class NullPlant:
    def __init__(self, params: NullParams):
        pass

    def step(self, state_x: State, command_u: ControlInput):
        return state_x, TableAccel(x_ddot=0.0, y_ddot=0.0)
