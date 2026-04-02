from src.shared import SystemState, TableAccel, TableCommand


class NullPlant:
    def __init__(self):
        pass

    def step(self, state_x: SystemState, command_u: TableCommand, dt):
        return state_x, TableAccel(x_ddot=0.0, y_ddot=0.0)
