import numpy as np
import control as ct

import argparse
from simulation_runner import run_simulation
from benchmark import region_mapping
from vision import VisionSystem
from estimator import FiniteDifferenceEstimator, KalmanEstimator, LowPassFiniteDifferenceEstimator
from visualization import Visualizer3D
from model import BuildLinearModel
from sim_types import (
    SystemState,
    PhysicalParams,
    CameraParams,
)



def main(mode="single"):

    params = PhysicalParams(
        g=9.81,
        com_length=0.1,
        tau=0.04,
        zeta=0.7,
        num_states=8
    )
    
    camera_params = CameraParams(xr=0.3, yr=0.3)
    vision = VisionSystem(camera_params, noise_std=0.01, delay_steps=10) # delay steps * dt = time delay
    estimator = LowPassFiniteDifferenceEstimator()
    
    '''
    # Kalman Filter
    Q = np.eye(8) * 1e-5
    R = np.eye(4) * 1e-4

    estimator = KalmanEstimator(A, dt=0.001, Q=Q, R=R)
    '''

    A, B = BuildLinearModel(params)

    desired_poles = [
        -8, -10, -12, -14,
        -8, -10, -12, -14
    ]

    K = ct.place(A, B, desired_poles)

    if mode == "single":

        initial_state = SystemState(
            x=0.0,
            x_dot=0.0,
            alpha_x=0.2,
            alpha_x_dot=0.0,
            y=0.0,
            y_dot=0.0,
            alpha_y=0.2,
            alpha_y_dot=0.0
        )

        result = run_simulation(
            params=params,
            initial_state=initial_state,
            total_time=5.0,
            dt=0.001,
            K=K,
            vision=vision,
            estimator=estimator
        )

        max_acc = np.max(np.abs(result.acc_history))
        print(f"Max table acceleration: {max_acc}")

        viz = Visualizer3D(result.state_history, dt=0.001)
        viz.render_video(video_speed=1, save_video=False)

    elif mode == "benchmark":

        results = region_mapping(
            params=params,
            K=K,
            dt=0.001,
            total_time=2.0,
            n_trials=200,
            vision=vision,
            estimator_class=FiniteDifferenceEstimator
        )
        
        stability_rate = sum(r["stabilized"] for r in results) / len(results)
        
        settling_times = [r["settling_time"] for r in results if r["stabilized"]]
        max_settling_time = max(settling_times) if settling_times else None
        avg_settling_time = sum(settling_times) / len(settling_times) if settling_times else None

        max_accs = [r["max_acc"] for r in results]
        max_acc_overall = max(max_accs)
        avg_acc = sum(max_accs) / len(max_accs)
        
        print(f"Stability rate: {stability_rate:.2%}")

        if settling_times:
            print(f"Max settling time: {max_settling_time:.3f} s")
            print(f"Avg settling time: {avg_settling_time:.3f} s")

        print(f"Max acceleration across trials: {max_acc_overall:.3f} m/s^2")
        print(f"Average peak acceleration: {avg_acc:.3f} m/s^2")

    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="single", choices=["single", "benchmark"])
    args = parser.parse_args()

    main(mode=args.mode)