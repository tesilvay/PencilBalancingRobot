import numpy as np
import control as ct
from core.sim_types import SystemState, PoseMeasurement, TableCommand

from perception.estimator_diagnostics import (
    EstimatorDiagnosticSnapshot,
    build_kalman_snapshot,
)
from perception.kalman_core import run_linear_kalman_step


# Variance on LPF velocity pseudo-measurements relative to pose variance (same σ² units per axis).
# Using the naive diff formula 2σ²/dt² (independent pose noise every step) makes R_vel enormous at
# dt=1 ms, so K_vel≈0 and the filter ignores z — prediction dominates and velocities blow up on
# hardware (stale z + model mismatch; see docs/kalman_real_world_gap.md). LPF rates are smoother
# than raw diff; scale off σ² so gains stay O(1) for typical P.


# -------------------------------------------------
# Base Interface
# -------------------------------------------------

class BaseEstimator:
    def __init__(self):
        self._last_diagnostic_snapshot: EstimatorDiagnosticSnapshot | None = None
        self._diag_step_idx = 0
        self._diag_t_s = 0.0
        self._diag_measurement_fresh = True
        self._diag_z_changed = True

    def set_diagnostics_context(
        self,
        step_idx: int,
        t_s: float,
        measurement_fresh: bool,
        z_changed: bool,
    ) -> None:
        self._diag_step_idx = step_idx
        self._diag_t_s = t_s
        self._diag_measurement_fresh = measurement_fresh
        self._diag_z_changed = z_changed

    def get_last_diagnostics(self) -> EstimatorDiagnosticSnapshot | None:
        return self._last_diagnostic_snapshot

    def update(
        self,
        pose: PoseMeasurement,
        dt: float,
        command_u: TableCommand | None = None,
    ) -> SystemState:
        raise NotImplementedError

    def reset(self):
        self._last_diagnostic_snapshot = None
    
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
            f"z:  "
            f"x={x*1000:+.2f} mm, x_dot={x_dot*1000:+.2f} mm/s, "
            f"ax={np.rad2deg(ax):+.2f}°, ax_dot={np.rad2deg(ax_dot):+.2f}°/s | "
            f"y={y*1000:+.2f} mm, y_dot={y_dot*1000:+.2f} mm/s, "
            f"ay={np.rad2deg(ay):+.2f}°, ay_dot={np.rad2deg(ay_dot):+.2f}°/s"
        )

    def _print_est_x_hat(self, est):
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


class FiniteDifferenceEstimator(BaseEstimator):

    def __init__(self):
        super().__init__()
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
        super().reset()
        self.prev_pose = None


class LowPassFiniteDifferenceEstimator(BaseEstimator):

    def __init__(self, alpha=0.93):
        super().__init__()
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
        super().reset()
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
        super().__init__()
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
        
        self.P_init = np.eye(8) * 2e-2 # bigger makes search more
        self.x_hat_init = np.zeros((8, 1))

        self.P = self.P_init
        #self.P = solve_discrete_are(A.T, self.H.T, self.Q, self.R)
        self.x_hat = self.x_hat_init

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

        step = run_linear_kalman_step(
            self.A,
            self.B,
            self.H,
            self.Q,
            self.R,
            self.x_hat,
            self.P,
            z,
            u,
        )
        self.x_hat = step.x_hat
        self.P = step.P
        self._last_diagnostic_snapshot = build_kalman_snapshot(
            estimator_name=type(self).__name__,
            step_idx=self._diag_step_idx,
            t_s=self._diag_t_s,
            dt_s=dt,
            measurement_fresh=self._diag_measurement_fresh,
            z_changed=self._diag_z_changed,
            step=step,
        )

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
        super().reset()
        self.P = self.P_init
        self.x_hat = self.x_hat_init


class FullStateKalmanFilter(BaseEstimator):
    """
    LPF finite-difference full state as measurement z ∈ R^8, fused with linear Kalman (H = I).
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        dt: float,
        Q: np.ndarray,
        R: np.ndarray,
        lpf: LowPassFiniteDifferenceEstimator,
    ):
        super().__init__()
        sys_c = ct.ss(A, B, np.eye(8), np.zeros((8, 2)))
        sys_d = ct.c2d(sys_c, dt)

        self.A = np.array(sys_d.A)
        self.B = np.array(sys_d.B)
        self.H = np.eye(8)
        self.Q = Q
        self.R = R
        self._lpf = lpf

        self.P = np.eye(8) * 0.01
        self.x_hat = np.zeros((8, 1))

    def update(
        self,
        pose: PoseMeasurement,
        dt: float,
        command_u: TableCommand | None = None,
    ) -> SystemState:
        z_state = self._lpf.update(pose, dt, command_u)
        z = z_state.as_vector().reshape(8, 1)

        if command_u is None:
            u = np.zeros((2, 1))
        else:
            u = np.array([[command_u.x_des], [command_u.y_des]])

        step = run_linear_kalman_step(
            self.A,
            self.B,
            self.H,
            self.Q,
            self.R,
            self.x_hat,
            self.P,
            z,
            u,
        )
        self.x_hat = step.x_hat
        self.P = step.P
        self._last_diagnostic_snapshot = build_kalman_snapshot(
            estimator_name=type(self).__name__,
            step_idx=self._diag_step_idx,
            t_s=self._diag_t_s,
            dt_s=dt,
            measurement_fresh=self._diag_measurement_fresh,
            z_changed=self._diag_z_changed,
            step=step,
        )

        return SystemState(
            x=self.x_hat[0, 0],
            x_dot=self.x_hat[1, 0],
            alpha_x=self.x_hat[2, 0],
            alpha_x_dot=self.x_hat[3, 0],
            y=self.x_hat[4, 0],
            y_dot=self.x_hat[5, 0],
            alpha_y=self.x_hat[6, 0],
            alpha_y_dot=self.x_hat[7, 0],
        )

    def reset(self):
        super().reset()
        self._lpf.reset()
        self.P = np.eye(8) * 0.01
        self.x_hat = np.zeros((8, 1))


