from sim_types import SystemState, TableCommand
import numpy as np
import control as ct


class BaseController:
    def compute(self, state):
        raise NotImplementedError
    
class PolePlacementController(BaseController):

    def __init__(self, A, B, desired_poles, x_ref=None):
        self.K = ct.place(A, B, desired_poles)
        self.x_ref = np.zeros(A.shape[0]) if x_ref is None else x_ref.as_vector()

    def compute(self, state):
        x_vec = state.as_vector()
        u = -self.K @ (x_vec - self.x_ref)
        return TableCommand(u[0], u[1])
    
class LQRController(BaseController):

    def __init__(self, A, B, Q, R, x_ref=None):
        self.K, _, _ = ct.lqr(A, B, Q, R)
        self.x_ref = np.zeros(A.shape[0]) if x_ref is None else x_ref.as_vector()

    def compute(self, state):
        x_vec = state.as_vector()
        u = -self.K @ (x_vec - self.x_ref)
        return TableCommand(u[0], u[1])

def build_lqr_weights(
    x_max,
    xdot_max,
    alpha_max,
    alphadot_max,
    u_max,
    angle_importance=1.0,
    effort_scale=0.1
):

#    Q_single_axis = np.diag([
#        1/x_max**2,
#        1/xdot_max**2,
#        angle_importance * (1/alpha_max**2),
#        angle_importance * (1/alphadot_max**2)
#    ])
    
    Q_single_axis = np.diag([10, 0.1, 10, 1]) # x, x_dot, alpha, alpha_dot

    # Symmetric block diagonal for x and y axes
    Q = np.block([
        [Q_single_axis, np.zeros((4,4))],
        [np.zeros((4,4)), Q_single_axis]
    ])

    R = np.eye(2) * effort_scale * (1/u_max**2)

    return Q, R


class NullController:
    def compute(self, state):
        # no actuation
        return TableCommand(0.0, 0.0)