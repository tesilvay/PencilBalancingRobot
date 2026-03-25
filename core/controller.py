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

class CircleController:
    def __init__(self, x_ref: SystemState, radius: float, period_s: float):
        self.x_ref = x_ref
        self.radius = radius
        self.period_s = period_s
        self.omega = 2 * np.pi / period_s
        self.t = 0.0

    def compute(self, state, dt):
        self.t += dt

        cx = self.x_ref.x
        cy = self.x_ref.y

        x = cx + self.radius * np.cos(self.omega * self.t)
        y = cy + self.radius * np.sin(self.omega * self.t)

        return TableCommand(x, y)

class NullController:
    def compute(self, state):
        # no actuation
        return TableCommand(0.0, 0.0)