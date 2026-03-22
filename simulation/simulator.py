from core.sim_types import SystemState, TableCommand, TableAccel
import numpy as np

class Simulator:

    def __init__(self, plant, controller, vision=None, estimator=None, dt=0.001, run_indefinitely: bool = False):
        self.plant = plant
        self.controller = controller
        self.vision = vision
        self.estimator = estimator
        self.dt = dt
        self.servo_timer = 0
        self.run_indefinitely = run_indefinitely

    def step(
        self,
        state_x_true: SystemState,
        command_u: TableCommand,
        realtime: bool,
        actuator_dt: float,
    ) -> tuple[SystemState, TableCommand, TableAccel, object, object]:

        if self.run_indefinitely:
            # No plant: use vision -> estimator -> controller. state_x_true is previous state_est.
            table_acc = TableAccel(x_ddot=0.0, y_ddot=0.0)
        else:
            # 1. Plant evolves using last command
            state_x_true, table_acc = self.plant.step(
                state_x_true,
                command_u,
                self.dt
            )

        # 2. Measurement
        if (self.vision is None) != (self.estimator is None):
            raise ValueError("Vision and estimator must both be set or both be None.")

        if self.vision is not None:
            measurement = self.vision.get_observation(state_x_true)
            if measurement is None:
                state_x_est = state_x_true
                pose = None
            else:
                pose = self.vision.reconstruct(measurement)
                state_x_est = self.estimator.update(
                    pose, self.dt, command_u
                )
        else:
            state_x_est = state_x_true
            measurement = None
            pose = None

        # 3. Controller at servo speed if we have one
        if self.servo_timer >= actuator_dt:
            command_u = self.controller.compute(state_x_est)
            self.servo_timer = 0

        self.servo_timer += self.dt

        # Real mode: return state_est (no true state). Sim: return state_x_true.
        state_out = state_x_est if self.run_indefinitely else state_x_true
        return state_out, command_u, table_acc, measurement, pose