from sim_types import SystemState, TableCommand

class Simulator:

    def __init__(self, plant, controller, dt=0.001):
        self.plant = plant
        self.controller = controller
        self.dt = dt

    def step(self, state_x: SystemState, command_u:TableCommand):

        # advance plant using last command
        state_x, table_acc = self.plant.step(
            state_x,
            command_u,
            self.dt
        )

        # compute new command from updated state
        command_u = self.controller.compute(state_x)

        return state_x, command_u, table_acc