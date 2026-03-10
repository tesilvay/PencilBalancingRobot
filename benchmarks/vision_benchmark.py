import numpy as np
from perception.vision import VisionSystem
from core.sim_types import CameraParams, SystemState

def compare_states(state_true, pose_est):

    print("----- Single Test -----")

    print(f"True X:        {state_true.x:.8f}")
    print(f"Estimated X:   {pose_est.X:.8f}")
    print(f"Abs Error X:   {abs(state_true.x - pose_est.X):.2e}")

    print()

    print(f"True Y:        {state_true.y:.8f}")
    print(f"Estimated Y:   {pose_est.Y:.8f}")
    print(f"Abs Error Y:   {abs(state_true.y - pose_est.Y):.2e}")

    print()

    print(f"True alpha_x:  {state_true.alpha_x:.8f}")
    print(f"Estimated ax:  {pose_est.alpha_x:.8f}")
    print(f"Abs Error ax:  {abs(state_true.alpha_x - pose_est.alpha_x):.2e}")

    print()

    print(f"True alpha_y:  {state_true.alpha_y:.8f}")
    print(f"Estimated ay:  {pose_est.alpha_y:.8f}")
    print(f"Abs Error ay:  {abs(state_true.alpha_y - pose_est.alpha_y):.2e}")

    print("-----------------------\n")


def monte_carlo_test(n_tests=1000):

    vision = VisionSystem(CameraParams(xr=0.3, yr=0.3))

    max_error = 0.0

    for i in range(n_tests):

        # Sample safely away from singularities
        X = np.random.uniform(-0.15, 0.15)
        Y = np.random.uniform(-0.15, 0.15)

        alpha_x = np.random.uniform(-0.4, 0.4)
        alpha_y = np.random.uniform(-0.4, 0.4)

        state_true = SystemState(
            x=X,
            x_dot=0.0,
            alpha_x=alpha_x,
            alpha_x_dot=0.0,
            y=Y,
            y_dot=0.0,
            alpha_y=alpha_y,
            alpha_y_dot=0.0
        )

        measurement = vision.project(state_true)
        pose_est = vision.reconstruct(measurement)

        errors = [
            abs(state_true.x - pose_est.X),
            abs(state_true.y - pose_est.Y),
            abs(state_true.alpha_x - pose_est.alpha_x),
            abs(state_true.alpha_y - pose_est.alpha_y)
        ]

        max_error = max(max_error, max(errors))

        if max(errors) > 1e-6:
            print("Large error detected!")
            compare_states(state_true, pose_est)
            break

    print(f"\nMonte Carlo finished.")
    print(f"Worst absolute error across {n_tests} runs: {max_error:.2e}")
    
    
state_true = SystemState(
    x=0.2,
    x_dot=0.3,
    alpha_x=0.1,
    alpha_x_dot=0.0,
    y=0.12,
    y_dot=0.0,
    alpha_y=0.23,
    alpha_y_dot=0.0
)

vision = VisionSystem(CameraParams(xr=0.3, yr=0.3))

measurement = vision.project(state_true)
pose_est = vision.reconstruct(measurement)

compare_states(state_true, pose_est)
monte_carlo_test(2000)