import numpy as np
from simulation_runner import run_simulation  # if needed for plant
from plant import BalancerPlant
from vision import VisionSystem
from model import BuildLinearModel
import matplotlib.pyplot as plt
from estimator import FiniteDifferenceEstimator, LowPassFiniteDifferenceEstimator, KalmanEstimator
from sim_types import (
    SystemState,
    PhysicalParams,
    CameraParams
)


# -------------------------------------------------
# Metric helpers
# -------------------------------------------------

def compute_rmse(true_history, est_history):
    error = true_history - est_history
    mse = np.mean(error**2, axis=0)
    return np.sqrt(mse)


# -------------------------------------------------
# Single estimator trial
# -------------------------------------------------

def run_estimator_trial(
    params,
    estimator,
    vision,
    total_time=3.0,
    dt=0.001
):

    plant = BalancerPlant(params)

    steps = int(total_time / dt)

    # Initial condition
    state_true = SystemState(
        x=0.0,
        x_dot=0.1,
        alpha_x=0.15,
        alpha_x_dot=0.1,
        y=0.0,
        y_dot=0.0,
        alpha_y=-0.12,
        alpha_y_dot=-0.15
    )

    true_history = np.zeros((steps, 8))
    est_history = np.zeros((steps, 8))

    for k in range(steps):

        t = k * dt

        # Smooth sinusoidal excitation
        x_des = 0.02 * np.sin(2*np.pi*0.5*t)
        y_des = 0.015 * np.cos(2*np.pi*0.3*t)

        # Plant evolves
        state_true, _ = plant.step(
            state_true,
            command_u=type("cmd", (), {"x_des": x_des, "y_des": y_des}),
            dt=dt
        )

        # Vision measurement
        measurement = vision.project(state_true)
        pose = vision.reconstruct(measurement)

        # Estimation
        state_est = estimator.update(pose, dt)
        
        if abs(state_true.alpha_x) > 0.5 or abs(state_true.alpha_y) > 0.5:
            true_history[k, :] = state_true.as_vector()
            est_history[k, :] = state_est.as_vector()
            break

    return true_history, est_history


# -------------------------------------------------
# Monte Carlo stress test
# -------------------------------------------------

def run_estimator_monte_carlo(
    params,
    estimator_class,
    camera_params,
    noise_std=None,
    delay_steps=0,
    n_trials=20,
    total_time=3.0,
    dt=0.001
):
    print("\n===================================")
    print(f"Estimator: {estimator_class.__name__}")
    print("===================================\n")
    
    rmses = []

    for i in range(n_trials):

        vision = VisionSystem(
            camera_params=camera_params,
            noise_std=noise_std,
            delay_steps=delay_steps
        )
        if estimator_class == KalmanEstimator: # it needs some info for initialization
            A, B = BuildLinearModel(params)
            Q = np.eye(8) * 1e-6  # small process noise
            R = np.eye(4) * 1e-4  # moderate measurement noise
            
            estimator = estimator_class(A=A, dt=0.001, Q=Q, R=R)
        else:
            estimator = estimator_class()

        true_hist, est_hist = run_estimator_trial(
            params=params,
            estimator=estimator,
            vision=vision,
            total_time=total_time,
            dt=dt
        )

        rmse = compute_rmse(true_hist, est_hist)
        rmses.append(rmse)
        print(f"Trial {i}: Velocity RMSE: x_dot = {rmse[1]:.4f}, alpha_x_dot = {rmse[3]:.4f}")

    rmses = np.array(rmses)

    avg_rmse = np.mean(rmses, axis=0)
    max_rmse = np.max(rmses, axis=0)

    state_labels = [
        "x (m)",
        "x_dot (m/s)",
        "alpha_x (rad)",
        "alpha_x_dot (rad/s)",
        "y (m)",
        "y_dot (m/s)",
        "alpha_y (rad)",
        "alpha_y_dot (rad/s)"
    ]

    print("\n===================================")
    print("Average RMSE per state:")
    print("===================================")

    for label, value in zip(state_labels, avg_rmse):
        print(f"{label:20s}: {value:.6e}")

    print("\n===================================")
    print("Worst-case RMSE per state:")
    print("===================================")

    for label, value in zip(state_labels, max_rmse):
        print(f"{label:20s}: {value:.6e}")

    return avg_rmse, max_rmse


def sweep_estimators(params, camera_params, dt=0.001):

    noise_levels = [0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2, 1e-2]
    delay_levels = [0, 1, 2, 3, 5, 10, 15, 50]

    estimators = [
        FiniteDifferenceEstimator,
        LowPassFiniteDifferenceEstimator,
        KalmanEstimator
    ]

    results = {}

    for estimator_class in estimators:

        print("\n===================================")
        print(f"Testing {estimator_class.__name__}")
        print("===================================")

        results[estimator_class.__name__] = []

        for noise in noise_levels:
            for delay in delay_levels:

                print(f"\nNoise={noise}, Delay={delay}")

                avg_rmse, _ = run_estimator_monte_carlo(
                    params=params,
                    estimator_class=estimator_class,
                    camera_params=camera_params,
                    noise_std=noise,
                    delay_steps=delay,
                    n_trials=5,     # reduce for speed
                    total_time=1.0, # shorter to prevent explosion
                    dt=dt
                )

                # store only alpha_x_dot RMSE (index 3)
                results[estimator_class.__name__].append({
                    "noise": noise,
                    "delay": delay,
                    "rmse_alpha_dot": avg_rmse[3]
                })

    return results, noise_levels, delay_levels


def plot_results(results, noise_levels, delay_levels):

    plt.figure()

    for name, data in results.items():

        rmse_vals = [d["rmse_alpha_dot"] for d in data]

        # reshape into matrix (noise x delay)
        rmse_matrix = np.array(rmse_vals).reshape(
            len(noise_levels),
            len(delay_levels)
        )

        # Plot worst delay case for each noise
        worst_delay_curve = rmse_matrix[:, -1]

        plt.plot(noise_levels, worst_delay_curve, marker='o', label=name)

    plt.xlabel("Measurement Noise Std")
    plt.ylabel("Alpha_x_dot RMSE (rad/s)")
    plt.title("Estimator Comparison (Worst Delay Case)")
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":

    camera_params = CameraParams(xr=0.3, yr=0.3)

    params = PhysicalParams(
        g=9.81,
        com_length=0.1,
        tau=0.04,
        zeta=0.7,
        num_states=8
    )

    results, noise_levels, delay_levels = sweep_estimators(
        params=params,
        camera_params=camera_params
    )

    plot_results(results, noise_levels, delay_levels)