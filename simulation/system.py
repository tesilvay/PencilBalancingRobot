from core.sim_types import WorkspaceParams, clamp_table_command_to_workspace
from numpy import rad2deg, array

class System:
    def __init__(self, plant, perception, dt, workspace: WorkspaceParams):
        self.plant = plant
        self.perception = perception
        self.dt = dt
        self.workspace = workspace
        self.SCALE = array([
            0.05,   # x (50 mm)
            0.5,    # x_dot (m/s)
            0.1,    # ax (rad)
            1.0,    # ax_dot (rad/s)
            0.05,   # y
            0.5,    # y_dot
            0.1,    # ay
            1.0     # ay_dot
        ])
    


    def _print_state(self, state):
        print(
            f"x={state.x*1000:+.2f} mm, x_dot={state.x_dot*1000:+.2f} mm/s, "
            f"ax={rad2deg(state.alpha_x):+.2f}°, ax_dot={rad2deg(state.alpha_x_dot):+.2f}°/s | "
            f"y={state.y*1000:+.2f} mm, y_dot={state.y_dot*1000:+.2f} mm/s, "
            f"ay={rad2deg(state.alpha_y):+.2f}°, ay_dot={rad2deg(state.alpha_y_dot):+.2f}°/s"
        )
        
    def _print_state_error(self, est, true):
        err = est - true
        norm_err = 100.0 * err / self.SCALE

        print(
            f"est err (% of scale): "
            f"x={norm_err[0]:+.1f}%, x_dot={norm_err[1]:+.1f}%, "
            f"ax={norm_err[2]:+.1f}%, ax_dot={norm_err[3]:+.1f}% | "
            f"y={norm_err[4]:+.1f}%, y_dot={norm_err[5]:+.1f}%, "
            f"ay={norm_err[6]:+.1f}%, ay_dot={norm_err[7]:+.1f}%"
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
        
        #self._print_state_error(state_est.as_vector(), state_true.as_vector())

        return state_true, state_est, acc, measurement, pose
