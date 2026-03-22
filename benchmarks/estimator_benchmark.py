# Estimator RMSE benchmark (open-loop falling pencil). Main's benchmark_single measures
# closed-loop stability with LQR, not estimation RMSE — results are not directly comparable.
import numpy as np
from collections import deque

from core.plant import BalancerPlant
from perception.vision import VisionModelBase, get_measurements
from core.model import BuildLinearModel
import matplotlib.pyplot as plt
from perception.estimator import (
    FiniteDifferenceEstimator,
    LowPassFiniteDifferenceEstimator,
    KalmanEstimator,
)
from core.sim_types import (
    SystemState,
    PhysicalParams,
    CameraParams,
    PlantParams,
    WorkspaceParams,
    TableCommand,
    CameraObservation,
    CameraPair,
)


def pencil_fell(state: SystemState) -> bool:
    return abs(state.alpha_x) > 2.5 or abs(state.alpha_y) > 2.5


def add_noise_to_camera_pair(cams: CameraPair, noise_std: float | None) -> CameraPair:
    """Same Gaussian noise on slopes/intercepts as SimVisionModel._add_noise."""
    b1, s1, b2, s2 = get_measurements(cams)
    if noise_std is not None:
        s1 += np.random.normal(0, noise_std)
        b1 += np.random.normal(0, noise_std)
        s2 += np.random.normal(0, noise_std)
        b2 += np.random.normal(0, noise_std)
    return CameraPair(
        cam1=CameraObservation(slope=s1, intercept=b1),
        cam2=CameraObservation(slope=s2, intercept=b2),
    )


def apply_noise_and_delay(
    noisy_cams: CameraPair,
    delay_steps: int,
    buffer: deque,
) -> CameraPair:
    """Match SimVisionModel.get_observation delay line (deque maxlen = delay_steps + 1)."""
    if delay_steps <= 0:
        return noisy_cams
    buffer.append(noisy_cams)
    if len(buffer) <= delay_steps:
        return noisy_cams
    return buffer[0]


# -------------------------------------------------
# Metric helpers
# -------------------------------------------------


def compute_rmse(true_history, est_history):
    error = true_history - est_history
    mse = np.mean(error**2, axis=0)
    return np.sqrt(mse)


def sample_initial_state(
    params: PhysicalParams,
    angle_spread_rad: float,
    ang_vel_spread_rad_s: float,
) -> SystemState:
    """Randomize only angles and angular rates; table at workspace reference, zero linear velocity."""
    w = params.workspace
    return SystemState(
        x=w.x_ref,
        x_dot=0.0,
        alpha_x=np.random.uniform(-angle_spread_rad, angle_spread_rad),
        alpha_x_dot=np.random.uniform(-ang_vel_spread_rad_s, ang_vel_spread_rad_s),
        y=w.y_ref,
        y_dot=0.0,
        alpha_y=np.random.uniform(-angle_spread_rad, angle_spread_rad),
        alpha_y_dot=np.random.uniform(-ang_vel_spread_rad_s, ang_vel_spread_rad_s),
    )


# -------------------------------------------------
# Single estimator trial
# -------------------------------------------------


def run_estimator_trial(
    params: PhysicalParams,
    estimator,
    camera_params: CameraParams,
    noise_std: float | None = None,
    delay_steps: int = 0,
    angle_spread_rad: float = 0.25,
    ang_vel_spread_rad_s: float = 0.5,
    total_time: float = 3.0,
    dt: float = 0.001,
    stop_on_fall: bool = True,
):
    """
    Open-loop falling pencil: constant table command at workspace reference.
    Vision: VisionModelBase.project + manual noise/delay + reconstruct.
    """
    plant = BalancerPlant(params)
    vision = VisionModelBase(camera_params)

    steps = int(total_time / dt)
    buffer: deque = deque(maxlen=delay_steps + 1) if delay_steps > 0 else deque()

    state_true = sample_initial_state(
        params, angle_spread_rad, ang_vel_spread_rad_s
    )

    w = params.workspace
    command = TableCommand(x_des=w.x_ref, y_des=w.y_ref)

    true_history = np.zeros((steps, 8))
    est_history = np.zeros((steps, 8))

    n_logged = steps
    for k in range(steps):
        state_true, _ = plant.step(state_true, command_u=command, dt=dt)

        raw = vision.project(state_true)
        noisy = add_noise_to_camera_pair(raw, noise_std)
        measurement = apply_noise_and_delay(noisy, delay_steps, buffer)
        pose = vision.reconstruct(measurement)

        state_est = estimator.update(pose, dt, command)

        true_history[k, :] = state_true.as_vector()
        est_history[k, :] = state_est.as_vector()

        if stop_on_fall and pencil_fell(state_true):
            n_logged = k + 1
            break

    return true_history[:n_logged], est_history[:n_logged]


# -------------------------------------------------
# Monte Carlo stress test
# -------------------------------------------------


def kalman_measurement_covariance(noise_std: float | None) -> np.ndarray:
    """Align with system_builder.build_estimator: R = sigma^2 I on pose channels."""
    if noise_std is None or noise_std <= 0:
        return np.eye(4) * 1e-8
    return np.eye(4) * (noise_std**2)


def run_estimator_monte_carlo(
    params: PhysicalParams,
    estimator_class,
    camera_params: CameraParams,
    noise_std: float | None = None,
    delay_steps: int = 0,
    n_trials: int = 20,
    total_time: float = 3.0,
    dt: float = 0.001,
    angle_spread_rad: float = 0.25,
    ang_vel_spread_rad_s: float = 0.5,
):
    rmses = []

    for _ in range(n_trials):
        if estimator_class == KalmanEstimator:
            A, B = BuildLinearModel(params)
            Q = np.eye(8) * 1e-6
            R = kalman_measurement_covariance(noise_std)
            estimator = estimator_class(A=A, B=B, dt=dt, Q=Q, R=R)
        elif estimator_class == LowPassFiniteDifferenceEstimator:
            estimator = estimator_class(alpha=0.95)
        else:
            estimator = estimator_class()

        true_hist, est_hist = run_estimator_trial(
            params=params,
            estimator=estimator,
            camera_params=camera_params,
            noise_std=noise_std,
            delay_steps=delay_steps,
            angle_spread_rad=angle_spread_rad,
            ang_vel_spread_rad_s=ang_vel_spread_rad_s,
            total_time=total_time,
            dt=dt,
        )

        rmse = compute_rmse(true_hist, est_hist)
        rmses.append(rmse)

    rmses = np.array(rmses)

    avg_rmse = np.mean(rmses, axis=0)
    max_rmse = np.max(rmses, axis=0)

    return avg_rmse, max_rmse


def sweep_estimators(params: PhysicalParams, camera_params: CameraParams, dt: float = 0.001):
    noise_levels = [0, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2, 5e-2, 1e-1]
    delay_levels = [0]

    estimators = [
        #FiniteDifferenceEstimator,
        LowPassFiniteDifferenceEstimator,
        KalmanEstimator,
    ]

    results = {}

    for estimator_class in estimators:
        print("\n===================================")
        print(f"Testing {estimator_class.__name__}")
        print("===================================")

        results[estimator_class.__name__] = []

        for noise in noise_levels:
            for delay in delay_levels:
                avg_rmse, _ = run_estimator_monte_carlo(
                    params=params,
                    estimator_class=estimator_class,
                    camera_params=camera_params,
                    noise_std=noise,
                    delay_steps=delay,
                    n_trials=5,
                    total_time=1.0,
                    dt=dt,
                )

                results[estimator_class.__name__].append(
                    {
                        "noise": noise,
                        "delay": delay,
                        "rmse_alpha_dot": avg_rmse[3],
                    }
                )

    return results, noise_levels, delay_levels


def plot_results(results, noise_levels, delay_levels):
    fig, ax = plt.subplots()

    # σ=0 is invalid on a log axis; place it slightly left of the smallest nonzero tick.
    noise_arr = np.asarray(noise_levels, dtype=float)
    min_pos = np.min(noise_arr[noise_arr > 0])
    x_plot = np.where(noise_arr > 0, noise_arr, min_pos * 0.2)

    for name, data in results.items():
        rmse_vals = [d["rmse_alpha_dot"] for d in data]

        rmse_matrix = np.array(rmse_vals).reshape(
            len(noise_levels),
            len(delay_levels),
        )

        worst_delay_curve = rmse_matrix[:, -1]

        ax.semilogx(x_plot, worst_delay_curve, marker="o", label=name)

    ax.set_xlabel("Measurement noise σ (std); σ=0 shown at leftmost point")
    ax.set_ylabel("Alpha_x_dot RMSE (rad/s)")
    ax.set_title("Estimator Comparison (Worst Delay Case)")
    ax.legend()
    ax.grid(True, which="major", ls="-", alpha=0.4)
    ax.grid(True, which="minor", ls=":", alpha=0.25)
    plt.show()


if __name__ == "__main__":
    # Align with main.py ExperimentSetup defaults
    camera_params = CameraParams(xr=0.170, yr=0.176)

    params = PhysicalParams(
        plant=PlantParams(
            g=9.81,
            com_length=0.1,
            tau=0.04,
            zeta=0.7,
            num_states=8,
            max_acc=9.81 * 3,
        ),
        workspace=WorkspaceParams(
            x_ref=0.0,
            y_ref=0.0,
            safe_radius=0.108,
        ),
    )

    results, noise_levels, delay_levels = sweep_estimators(
        params=params,
        camera_params=camera_params,
    )

    plot_results(results, noise_levels, delay_levels)
