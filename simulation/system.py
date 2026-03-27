from core.sim_types import WorkspaceParams, clamp_table_command_to_workspace


class System:
    def __init__(self, plant, perception, controller, dt, workspace: WorkspaceParams):
        self.plant = plant
        self.perception = perception
        self.controller = controller
        self.dt = dt
        self.workspace = workspace

    def step(self, state, command):

        state_true, acc = self.plant.step(state, command, self.dt)

        if self.perception:
            state_est, measurement, pose = self.perception.update(state_true, command, self.dt)
        else:
            state_est, measurement, pose = state_true, None, None

        u_raw = self.controller.compute(state_est)
        command = clamp_table_command_to_workspace(u_raw, self.workspace)

        return state_true, state_est, command, acc, measurement, pose
