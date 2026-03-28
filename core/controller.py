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


class SmoothPolePlacementController(BaseController):
    """Discrete-time Δu feedback via augmented state ξ = [x; u_{k-1}], v = Δu, gains from pole placement."""

    def __init__(
        self,
        A_c: np.ndarray,
        B_c: np.ndarray,
        dt: float,
        desired_poles_z: np.ndarray,
        x_ref: SystemState | None = None,
    ):
        n, m = A_c.shape[0], B_c.shape[1]
        sys_c = ct.ss(A_c, B_c, np.eye(n), np.zeros((n, m)))
        sys_d = ct.c2d(sys_c, dt)
        A_d = np.array(sys_d.A)
        B_d = np.array(sys_d.B)

        z = np.asarray(desired_poles_z, dtype=complex).ravel()
        if z.size != n + m:
            raise ValueError(
                f"desired_poles_z must have length {n + m} (dim ξ), got {z.size}"
            )

        A_aug = np.block([[A_d, B_d], [np.zeros((m, n)), np.eye(m)]])
        B_aug = np.vstack([B_d, np.eye(m)])
        self.K = ct.place(A_aug, B_aug, z)

        self.x_ref = np.zeros(n) if x_ref is None else x_ref.as_vector()
        self.u_ref = (-np.linalg.pinv(B_c) @ (A_c @ self.x_ref)).ravel()
        self.xi_ref = np.concatenate([self.x_ref, self.u_ref])
        self._u_prev = self.u_ref.copy()

    def compute(self, state: SystemState) -> TableCommand:
        x = state.as_vector()
        xi = np.concatenate([x, self._u_prev])
        v = -(self.K @ (xi - self.xi_ref)).ravel()
        u = self._u_prev + v
        return TableCommand(float(u[0]), float(u[1]))

    def set_applied_command(self, cmd: TableCommand) -> None:
        self._u_prev = np.array([cmd.x_des, cmd.y_des], dtype=float)


class CircleController:
    def __init__(self, x_ref: SystemState, radius: float, period_s: float):
        self.x_ref = x_ref
        self.radius = radius
        self.period_s = period_s
        self.omega = 2 * np.pi / period_s
        self.t = 0.0
        self.dt = 0.001

    def compute(self, state):
        self.t += self.dt

        cx = self.x_ref.x
        cy = self.x_ref.y

        x = cx + self.radius * np.cos(self.omega * self.t)
        y = cy + self.radius * np.sin(self.omega * self.t)

        return TableCommand(x, y)

class NullController:
    def compute(self, state):
        # no actuation
        return TableCommand(0.0, 0.0)