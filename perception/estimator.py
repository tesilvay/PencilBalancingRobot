import numpy as np
import control as ct
from core.sim_types import SystemState, PoseMeasurement, TableCommand
from scipy.linalg import solve_discrete_are


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
        raise NotImplementedError


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
            vel = np.zeros(4)
        else:
            vel = np.array([
                (pose.X - self.prev_pose.X) / dt,
                (pose.alpha_x - self.prev_pose.alpha_x) / dt,
                (pose.Y - self.prev_pose.Y) / dt,
                (pose.alpha_y - self.prev_pose.alpha_y) / dt
            ])

        self.prev_pose = pose

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


class LowPassFiniteDifferenceEstimator(BaseEstimator):

    def __init__(self, alpha=0.95):
        self.prev_pose = None
        self.prev_vel = np.zeros(4)
        self.alpha = 0.95 if alpha is None else alpha

    def update(
        self,
        pose: PoseMeasurement,
        dt: float,
        command_u: TableCommand | None = None,
    ) -> SystemState:

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

        self.P = np.eye(8) * 0.01 #guess
        #self.P = solve_discrete_are(A.T, self.H.T, self.Q, self.R)
        self.x_hat = np.zeros((8, 1))


    def _print_state(self, state):
        print(
            f"x={state.x*1000:+.2f} mm, x_dot={state.x_dot*1000:+.2f} mm/s, "
            f"ax={np.rad2deg(state.alpha_x):+.2f}°, ax_dot={np.rad2deg(state.alpha_x_dot):+.2f}°/s | "
            f"y={state.y*1000:+.2f} mm, y_dot={state.y_dot*1000:+.2f} mm/s, "
            f"ay={np.rad2deg(state.alpha_y):+.2f}°, ay_dot={np.rad2deg(state.alpha_y_dot):+.2f}°/s"
        )
        
    def _print_vel(self, state):
        print(
            f"x_dot={state.x_dot*1000:+.2f} mm/s, "
            f"ax_dot={np.rad2deg(state.alpha_x_dot):+.2f}°/s | "
            f"y_dot={state.y_dot*1000:+.2f} mm/s, "
            f"ay_dot={np.rad2deg(state.alpha_y_dot):+.2f}°/s"
        )
    
    def _print_pose(self, pose):
        x = pose[0, 0]
        ax = pose[1, 0]
        y = pose[2, 0]
        ay = pose[3, 0]
        print(
        f"pose:   "
        f"x={x*1000:+.2f} mm, "
        f"ax={np.rad2deg(ax):+.2f}° | "
        f"y={y*1000:+.2f} mm, "
        f"ay={np.rad2deg(ay):+.2f}°"
        )
    
    def _print_est(self, est):
        x = est[0, 0]
        x_dot = est[1, 0]
        ax = est[2, 0]
        ax_dot = est[3, 0]
        y = est[4, 0]
        y_dot = est[5, 0]
        ay = est[6, 0]
        ay_dot = est[7, 0]
        
        print(
            f"x_hat:  "
            f"x={x*1000:+.2f} mm, x_dot={x_dot*1000:+.2f} mm/s, "
            f"ax={np.rad2deg(ax):+.2f}°, ax_dot={np.rad2deg(ax_dot):+.2f}°/s | "
            f"y={y*1000:+.2f} mm, y_dot={y_dot*1000:+.2f} mm/s, "
            f"ay={np.rad2deg(ay):+.2f}°, ay_dot={np.rad2deg(ay_dot):+.2f}°/s"
        )
    

    def update(
        self,
        pose: PoseMeasurement,
        dt: float,
        command_u: TableCommand | None = None,
    ) -> SystemState:

        z = np.array(
            [pose.X, pose.alpha_x, pose.Y, pose.alpha_y], dtype=float
        ).reshape(-1, 1)
        
        self._print_pose(z)

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
        
        self._print_est(self.x_hat)

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



