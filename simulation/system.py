class System:
    def __init__(self, plant, perception, controller, dt):
        self.plant = plant
        self.perception = perception
        self.controller = controller
        self.dt = dt

    def step(self, state, command):

        state_true, acc = self.plant.step(state, command, self.dt)

        if self.perception:
            state_est, measurement, pose = self.perception.update(state_true, command, self.dt)
        else:
            state_est, measurement, pose = state_true, None, None

        command = self.controller.compute(state_est)

        return state_true, state_est, command, acc, measurement, pose
