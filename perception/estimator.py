import numpy as np
import control as ct
from core.sim_types import SystemState, PoseMeasurement, TableCommand


# -------------------------------------------------
# Base Interface
# -------------------------------------------------

class BaseEstimator:
    def update(
        self,
        pose: PoseMeasurement,
        dt: float,
        command_u: TableCommand | None = None,
    ) -> SystemState:
        raise NotImplementedError

    def reset(self):
        pass


class FiniteDifferenceEstimator(BaseEstimator):

    def __init__(self):
        self.prev_pose = None

    def update(
        self,
        pose: PoseMeasurement,
        dt: float,
        command_u: TableCommand | None = None,
    ) -> SystemState:

        if self.prev_pose is None:
            # First call: no velocity info
            x_dot = 0.0
            y_dot = 0.0
            alpha_x_dot = 0.0
            alpha_y_dot = 0.0
        else:
            x_dot = (pose.X - self.prev_pose.X) / dt
            y_dot = (pose.Y - self.prev_pose.Y) / dt
            alpha_x_dot = (pose.alpha_x - self.prev_pose.alpha_x) / dt
            alpha_y_dot = (pose.alpha_y - self.prev_pose.alpha_y) / dt

        self.prev_pose = pose

        return SystemState(
            x=pose.X,
            x_dot=x_dot,
            alpha_x=pose.alpha_x,
            alpha_x_dot=alpha_x_dot,
            y=pose.Y,
            y_dot=y_dot,
            alpha_y=pose.alpha_y,
            alpha_y_dot=alpha_y_dot
        )
    
    def reset(self):
        self.prev_pose = None


class LowPassFiniteDifferenceEstimator(BaseEstimator):

    def __init__(self, alpha=0.95):
        self.prev_pose = None
        self.prev_vel = np.zeros(4)
        self.alpha = 0.95 if alpha is None else alpha

    def update(self, pose, dt, command_u: TableCommand | None = None):

        if self.prev_pose is None:
            vel = np.zeros(4)
        else:
            raw_vel = np.array([
                (pose.X - self.prev_pose.X) / dt,
                (pose.alpha_x - self.prev_pose.alpha_x) / dt,
                (pose.Y - self.prev_pose.Y) / dt,
                (pose.alpha_y - self.prev_pose.alpha_y) / dt
            ])

            vel = self.alpha * self.prev_vel + (1 - self.alpha) * raw_vel

        self.prev_pose = pose
        self.prev_vel = vel

        return SystemState(
            x=pose.X,
            x_dot=vel[0],
            alpha_x=pose.alpha_x,
            alpha_x_dot=vel[1],
            y=pose.Y,
            y_dot=vel[2],
            alpha_y=pose.alpha_y,
            alpha_y_dot=vel[3]
        )

    def reset(self):
        self.prev_pose = None
        self.prev_vel = np.zeros(4)


class KalmanEstimator(BaseEstimator):

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        dt: float,
        Q: np.ndarray,
        R: np.ndarray,
    ):

        # Discretize continuous system with control input u = [x_des, y_des]
        sys_c = ct.ss(A, B, np.eye(8), np.zeros((8, 2)))
        sys_d = ct.c2d(sys_c, dt)

        self.A = np.array(sys_d.A)
        self.B = np.array(sys_d.B)

        # Measurement matrix
        # z = [X, alpha_x, Y, alpha_y]
        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1.0  # X
        self.H[1, 2] = 1.0  # alpha_x
        self.H[2, 4] = 1.0  # Y
        self.H[3, 6] = 1.0  # alpha_y

        self.Q = Q
        self.R = R

        self.P = np.eye(8) * 0.01
        self.x_hat = np.zeros((8, 1))

    def update(
        self,
        pose: PoseMeasurement,
        dt: float,
        command_u: TableCommand | None = None,
    ) -> SystemState:

        z = np.array(
            [pose.X, pose.alpha_x, pose.Y, pose.alpha_y], dtype=float
        ).reshape(-1, 1)

        if command_u is None:
            u = np.zeros((2, 1))
        else:
            u = np.array([[command_u.x_des], [command_u.y_des]])

        # ----- Prediction -----
        x_pred = self.A @ self.x_hat + self.B @ u
        P_pred = self.A @ self.P @ self.A.T + self.Q

        # ----- Update -----
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        y = z - self.H @ x_pred

        self.x_hat = x_pred + K @ y
        self.P = (np.eye(8) - K @ self.H) @ P_pred

        return SystemState(
            x=self.x_hat[0, 0],
            x_dot=self.x_hat[1, 0],
            alpha_x=self.x_hat[2, 0],
            alpha_x_dot=self.x_hat[3, 0],
            y=self.x_hat[4, 0],
            y_dot=self.x_hat[5, 0],
            alpha_y=self.x_hat[6, 0],
            alpha_y_dot=self.x_hat[7, 0]
        )

    def reset(self):
        self.P = np.eye(8) * 0.01
        self.x_hat = np.zeros((8, 1))



