from core.sim_types import SystemState, TableCommand, TableAccel
import numpy as np

class Simulator:

    def __init__(self, plant, controller, vision=None, estimator=None, dt=0.001):
        self.plant = plant
        self.controller = controller
        self.vision = vision
        self.estimator=estimator
        self.dt = dt
        self.servo_timer=0

    def step(
        self,
        state_x_true: SystemState,
        command_u: TableCommand,
        realtime: bool,
        actuator_dt: float,
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
        
        if self.vision is not None:
            measurement = self.vision.get_observation(state_x_true)
            if measurement is None:
                # estimator cannot update yet, use ground truth. we should estimate instead
                state_x_est = state_x_true
                pose = None
            else:
                pose = self.vision.reconstruct(measurement)
                state_x_est = self.estimator.update(pose, self.dt)
        else:
            state_x_est = state_x_true
            measurement = None
            pose = None

        # 3. Controller at servo speed if we have one
       
        if self.servo_timer >= actuator_dt:
            command_u = self.controller.compute(state_x_est)
            self.servo_timer = 0

        self.servo_timer += self.dt


        return state_x_true, command_u, table_acc, measurement, pose