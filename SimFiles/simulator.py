from sim_types import SystemState, TableCommand, TableAccel
import numpy as np

class Simulator:

    def __init__(self, plant, controller, vision=None, estimator=None, dt=0.001):
        self.plant = plant
        self.controller = controller
        self.vision = vision
        self.estimator=estimator
        self.dt = dt

    def step(
        self,
        state_x_true: SystemState,
        command_u: TableCommand
    ) -> tuple[SystemState, TableCommand, TableAccel]:

        # 1. Plant evolves using last command
        state_x_true, table_acc = self.plant.step(
            state_x_true,
            command_u,
            self.dt
        )
        
        # 2. Measurement 
        
        # Enforce configuration
        if (self.vision is None) != (self.estimator is None):
            raise ValueError("Vision and estimator must both be set or both be None.")
        
        if self.vision is not None: # estimate states
            # Vision measures pose
            # Estimator measures full state (calculates derivatives, which vision can't do)
            measurement = self.vision.project(state_x_true) # Generates camera povs
            pose = self.vision.reconstruct(measurement)     # Reconstructs pose
            state_x_est = self.estimator.update(pose, self.dt)       # Estimates state variables
        else: # use ground truth
            state_x_est = state_x_true

        # 3. Controller uses estimate
        command_u = self.controller.compute(state_x_est)

        return state_x_true, command_u, table_acc