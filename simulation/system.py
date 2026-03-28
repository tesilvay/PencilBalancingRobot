from core.sim_types import WorkspaceParams, clamp_table_command_to_workspace
from numpy import rad2deg

class System:
    def __init__(self, plant, perception, controller, dt, workspace: WorkspaceParams):
        self.plant = plant
        self.perception = perception
        self.controller = controller
        self.dt = dt
        self.workspace = workspace
    


    def _print_state(self, state):
        print(
            f"x={state.x*1000:+.2f} mm, x_dot={state.x_dot*1000:+.2f} mm/s, "
            f"ax={rad2deg(state.alpha_x):+.2f}°, ax_dot={rad2deg(state.alpha_x_dot):+.2f}°/s | "
            f"y={state.y*1000:+.2f} mm, y_dot={state.y_dot*1000:+.2f} mm/s, "
            f"ay={rad2deg(state.alpha_y):+.2f}°, ay_dot={rad2deg(state.alpha_y_dot):+.2f}°/s"
        )
        
    def _print_vel(self, state):
        print(
            f"x_dot={state.x_dot*1000:+.2f} mm/s, "
            f"ax_dot={rad2deg(state.alpha_x_dot):+.2f}°/s | "
            f"y_dot={state.y_dot*1000:+.2f} mm/s, "
            f"ay_dot={rad2deg(state.alpha_y_dot):+.2f}°/s"
        )

    def step(self, state, command):

        state_true, acc = self.plant.step(state, command, self.dt)

        if self.perception:
            state_est, measurement, pose = self.perception.update(state_true, command, self.dt)
        else:
            state_est, measurement, pose = state_true, None, None
        
        #self._print_state(state_est)

        u_raw = self.controller.compute(state_est)
        command = clamp_table_command_to_workspace(u_raw, self.workspace)

        return state_true, state_est, command, acc, measurement, pose
