from src.shared import NullParams, SystemState, TableAccel, TableCommand


class NullPlant:
    def __init__(self, params: NullParams):
        pass

    def step(self, state_x: SystemState, command_u: TableCommand, dt):
        return state_x, TableAccel(x_ddot=0.0, y_ddot=0.0)
