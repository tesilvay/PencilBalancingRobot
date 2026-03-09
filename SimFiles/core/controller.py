from core.sim_types import SystemState, TableCommand
import numpy as np
import control as ct


class BaseController:
    def compute(self, state):
        raise NotImplementedError
    
class PolePlacementController(BaseController):

    def __init__(self, A, B, desired_poles, x_ref=None):
        self.K = ct.place(A, B, desired_poles)

        self.x_ref = np.zeros(A.shape[0]) if x_ref is None else x_ref.as_vector()

        # compute steady-state feedforward
        self.u_ref = -np.linalg.pinv(B) @ (A @ self.x_ref)

    def compute(self, state):
        x = state.as_vector()
        
        error = x - self.x_ref

        u = self.u_ref - self.K @ (error)

        return TableCommand(u[0], u[1])
    
class LQRController(BaseController):

    def __init__(self, A, B, Q, R, x_ref=None):
        self.K, _, _ = ct.lqr(A, B, Q, R)
        self.x_ref = np.zeros(A.shape[0]) if x_ref is None else x_ref.as_vector()
        
        # compute steady-state feedforward
        self.u_ref = -np.linalg.pinv(B) @ (A @ self.x_ref)

    def compute(self, state):
        x = state.as_vector()
        
        error = x - self.x_ref

        u = self.u_ref - self.K @ (error)

        return TableCommand(u[0], u[1])


class NullController:
    def compute(self, state):
        # no actuation
        return TableCommand(0.0, 0.0)