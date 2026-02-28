from sim_types import SystemState, TableCommand
import numpy as np

class StateFeedbackController:
    def __init__(self, K: np.ndarray):
        """
        K must be shape (2, 8)
        First row -> x_des
        Second row -> y_des
        """
        self.K = K

    def compute(self, state_x: SystemState) -> TableCommand:
        x_vec = state_x.as_vector()
        u = -self.K @ x_vec
        
        return TableCommand(
            x_des=u[0],
            y_des=u[1]
        )