"""
Benchmark Hough line tracking: static lines and falling pencil.

- run_hough_monte_carlo: static line (original) - how well does Hough track a fixed line?
- run_falling_pencil_benchmark: falling pencil, no controller - how close does Hough
  get to true state as the pencil falls? Monte Carlo over initial conditions.
  This verifies perception quality and lag under dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

from perception.dvs_algorithms import PaperHoughLineAlgorithm
from perception.camera_model import CameraModel
from perception.vision import VisionModelBase, get_measurements
from core.sim_types import (
    CameraObservation,
    CameraPair,
    SystemState,
    TableCommand,
    PhysicalParams,
    PlantParams,
    WorkspaceParams,
    MechanismParams,
    HardwareParams,
    RunParams,
    CameraParams,
)
from core.plant import BalancerPlant


# -------------------------------------------------
# Generate events from a ground truth line
# -------------------------------------------------

def generate_events(cam, obs: CameraObservation, n=500, noise_px=1.0):

    obs_px = cam.normalized_to_pixel(obs)

    s_px = obs_px.slope
    b_px = obs_px.intercept

    # Sample uniform in y for x = s*y + b, then filter to valid pixel bounds.
    ys = np.random.uniform(0, cam.height, n)
    xs = s_px * ys + b_px

    xs += np.random.normal(0, noise_px, n)
    ys += np.random.normal(0, noise_px, n)

    mask = (xs >= 0) & (xs < cam.width) & (ys >= 0) & (ys < cam.height)
    xs = xs[mask]
    ys = ys[mask]

    xs = xs.astype(np.int16)
    ys = ys.astype(np.int16)

    events = np.zeros(len(xs), dtype=[("x", np.int16), ("y", np.int16)])
    events["x"] = xs
    events["y"] = ys

    return events


# -------------------------------------------------
# Single trial (static line - original)
# -------------------------------------------------

def run_hough_trial(
    obs_true: CameraObservation,
    steps=100,
    noise_px=1.0
):

    cam = CameraModel()
    algo = PaperHoughLineAlgorithm()

    b_est_hist = []
    s_est_hist = []

    for k in range(steps):

        events = generate_events(cam, obs_true, noise_px=noise_px)

        obs_px = algo.update(events)

        if isinstance(obs_px, tuple):
            continue

        # convert pixel line → normalized line
        obs_est = cam.pixel_to_normalized(obs_px)

        b_est_hist.append(obs_est.intercept)
        s_est_hist.append(obs_est.slope)

    return np.array(b_est_hist), np.array(s_est_hist)


# -------------------------------------------------
# Monte Carlo test (static line - original)
# -------------------------------------------------

def run_hough_monte_carlo(
    n_trials=20,
    noise_px=1.0
):

    errors_b = []
    errors_s = []

    for i in range(n_trials):

        # random ground truth
        b_true = np.random.uniform(-0.3, 0.3)
        s_true = np.random.uniform(-0.2, 0.2)

        obs_true = CameraObservation(slope=s_true, intercept=b_true)

        b_est_hist, s_est_hist = run_hough_trial(
            obs_true=obs_true,
            noise_px=noise_px
        )

        if len(b_est_hist) == 0:
            continue

        b_error = np.mean(np.abs(b_est_hist - obs_true.intercept))
        s_error = np.mean(np.abs(s_est_hist - obs_true.slope))

        errors_b.append(b_error)
        errors_s.append(s_error)

        print(f"Trial {i}: slope err={s_error:.6f} intercept err={b_error:.6f}")

    print("\n==== Summary ====")

    print("Mean slope error:", np.mean(errors_s))
    print("Mean intercept error:", np.mean(errors_b))


# -------------------------------------------------
# Falling pencil benchmark: Hough vs true state
# -------------------------------------------------

def _make_params():
    """Same params as main.py for consistency."""
    return PhysicalParams(
        plant=PlantParams(
            g=9.81,
            com_length=0.1,
            tau=0.02,
            zeta=0.7,
            num_states=8,
            max_acc=9.81 * 9,
        ),
        workspace=WorkspaceParams(
            x_ref=-0.00993,
            y_ref=0.01553,
            safe_radius=0.040,
        ),
        mechanism=MechanismParams(
            O=(83, 57),
            B=(61, 88),
            la=77,
            lb=69.6,
        ),
        hardware=HardwareParams(),
        run=RunParams(),
    )


def run_falling_pencil_trial(
    params: PhysicalParams,
    camera_params: CameraParams,
    initial_state: SystemState,
    total_time: float = 0.5,
    dt: float = 0.001,
    n_events_per_step: int = 200,
    noise_px: float = 1.0,
    hough_decay: float = 0.999,
):
    """
    Simulate falling pencil (no controller). Table held at ref.
    At each step: true state → project → events → Hough → compare.
    Returns (true_pose_history, hough_pose_history, error_history).
    """
    plant = BalancerPlant(params)
    vision_base = VisionModelBase(camera_params)
    cam = CameraModel()
    cam1_algo = PaperHoughLineAlgorithm(decay=hough_decay)
    cam2_algo = PaperHoughLineAlgorithm(decay=hough_decay)

    x_ref = params.workspace.x_ref
    y_ref = params.workspace.y_ref
    command = TableCommand(x_ref, y_ref)

    steps = int(total_time / dt)

    true_poses = []   # (X, Y, alpha_x, alpha_y)
    hough_poses = []  # (X, Y, alpha_x, alpha_y) or None
    state = initial_state

    for i in range(steps):
        # 1. Plant step (no controller - table follows ref)
        state, _ = plant.step(state, command, dt)

        # 2. True pose from state
        cams_true = vision_base.project(state)
        pose_true = vision_base.reconstruct(cams_true)
        true_poses.append((pose_true.X, pose_true.Y, pose_true.alpha_x, pose_true.alpha_y))

        # 3. Generate events from true lines (same as SimEventCameraInterface)
        b1, s1, b2, s2 = get_measurements(cams_true)
        events1 = _generate_events(cam, b1, s1, n=n_events_per_step, noise_px=noise_px)
        events2 = _generate_events(cam, b2, s2, n=n_events_per_step, noise_px=noise_px)

        # 4. Hough update
        result1 = cam1_algo.update(events1)
        result2 = cam2_algo.update(events2)

        if isinstance(result1, tuple) or isinstance(result2, tuple):
            hough_poses.append(None)
            continue

        obs1 = cam.pixel_to_normalized(result1)
        obs2 = cam.pixel_to_normalized(result2)
        cams_hough = CameraPair(
            CameraObservation(slope=obs1.slope, intercept=obs1.intercept),
            CameraObservation(slope=obs2.slope, intercept=obs2.intercept),
        )
        pose_hough = vision_base.reconstruct(cams_hough)
        hough_poses.append((pose_hough.X, pose_hough.Y, pose_hough.alpha_x, pose_hough.alpha_y))

    return np.array(true_poses), hough_poses


def _generate_events(cam, b, s, n=200, noise_px=1.0):
    """Same as SimEventCameraInterface.generate_events (normalized coords)."""
    obs_px = cam.normalized_to_pixel(CameraObservation(slope=s, intercept=b))
    s_px, b_px = obs_px.slope, obs_px.intercept

    ys = np.random.uniform(0, cam.height, n)
    xs = s_px * ys + b_px
    xs += np.random.normal(0, noise_px, n)

    mask = (xs >= 0) & (xs < cam.width) & (ys >= 0) & (ys < cam.height)
    xs = xs[mask]
    ys = ys[mask]

    xs = xs.astype(np.int16)
    ys = ys.astype(np.int16)

    events = np.zeros(len(xs), dtype=[("x", np.int16), ("y", np.int16)])
    events["x"] = xs
    events["y"] = ys
    return events


def run_falling_pencil_benchmark(
    n_trials: int = 50,
    total_time: float = 0.5,
    alpha_max: float = 0.05,
    hough_decay: float = 0.999,
    show_progress: bool = True,
):
    """
    Monte Carlo benchmark: falling pencil, no controller.
    Measures how close Hough perception gets to true state.
    """
    params = _make_params()
    camera_params = CameraParams(xr=0.3, yr=0.3)
    x_ref = params.workspace.x_ref
    y_ref = params.workspace.y_ref
    dt = 0.001

    all_errors_X = []
    all_errors_Y = []
    all_errors_alpha_x = []
    all_errors_alpha_y = []
    valid_steps_per_trial = []

    for trial in range(n_trials):
        if show_progress:
            pct = 100 * (trial + 1) / n_trials
            sys.stdout.write(f"\rFalling pencil benchmark: {pct:.0f}% ({trial+1}/{n_trials})")
            sys.stdout.flush()

        initial_state = SystemState(
            x=x_ref,
            x_dot=0.0,
            alpha_x=np.random.uniform(-alpha_max, alpha_max),
            alpha_x_dot=0.0,
            y=y_ref,
            y_dot=0.0,
            alpha_y=np.random.uniform(-alpha_max, alpha_max),
            alpha_y_dot=0.0,
        )

        true_poses, hough_poses = run_falling_pencil_trial(
            params=params,
            camera_params=camera_params,
            initial_state=initial_state,
            total_time=total_time,
            dt=dt,
            hough_decay=hough_decay,
        )

        # Compute errors over valid steps
        err_X, err_Y, err_ax, err_ay = [], [], [], []
        for step, hough in enumerate(hough_poses):
            if hough is None:
                continue
            X_t, Y_t, ax_t, ay_t = true_poses[step]
            X_h, Y_h, ax_h, ay_h = hough
            err_X.append(abs(X_h - X_t))
            err_Y.append(abs(Y_h - Y_t))
            err_ax.append(abs(ax_h - ax_t))
            err_ay.append(abs(ay_h - ay_t))

        if err_X:
            all_errors_X.extend(err_X)
            all_errors_Y.extend(err_Y)
            all_errors_alpha_x.extend(err_ax)
            all_errors_alpha_y.extend(err_ay)
            valid_steps_per_trial.append(len(err_X))

    if show_progress:
        sys.stdout.write("\n")

    # Summary
    n_valid = len(all_errors_X)
    if n_valid == 0:
        print("No valid Hough estimates across trials.")
        return

    print("\n==== Falling Pencil Benchmark (no controller) ====")
    print(f"Trials: {n_trials}, total_time: {total_time}s, dt: {dt}s")
    print(f"Initial alpha range: ±{alpha_max} rad, Hough decay: {hough_decay}")
    print(f"Valid Hough estimates: {n_valid}")

    print("\n--- Mean absolute error (Hough vs true) ---")
    print(f"  X (m):        {np.mean(all_errors_X):.6f}")
    print(f"  Y (m):        {np.mean(all_errors_Y):.6f}")
    print(f"  alpha_x (r):  {np.mean(all_errors_alpha_x):.6f}")
    print(f"  alpha_y (r):  {np.mean(all_errors_alpha_y):.6f}")
    mean_alpha_err = np.mean(all_errors_alpha_x + all_errors_alpha_y)
    print(f"  alpha (deg):  {np.rad2deg(mean_alpha_err):.4f}")

    print("\n--- Percentiles (alpha error, rad) ---")
    all_alpha_err = all_errors_alpha_x + all_errors_alpha_y
    for p in [50, 90, 95, 99]:
        print(f"  p{p}: {np.percentile(all_alpha_err, p):.6f} rad ({np.rad2deg(np.percentile(all_alpha_err, p)):.4f} deg)")

    print("\n--- Mean valid steps per trial ---")
    print(f"  {np.mean(valid_steps_per_trial):.1f} / {int(total_time / dt)}")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["static", "falling", "both", "decay_sweep"], default="falling")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--total_time", type=float, default=0.5)
    parser.add_argument("--alpha_max", type=float, default=0.05)
    parser.add_argument("--decay", type=float, default=0.999, help="Hough decay (0.999=laggy, 0.99=responsive)")
    args = parser.parse_args()

    if args.mode in ("static", "both"):
        print("\n=== Static line benchmark ===\n")
        run_hough_monte_carlo(n_trials=20, noise_px=1.0)

    if args.mode in ("falling", "both"):
        print("\n=== Falling pencil benchmark ===\n")
        run_falling_pencil_benchmark(
            n_trials=args.n_trials,
            total_time=args.total_time,
            alpha_max=args.alpha_max,
            hough_decay=args.decay,
        )

    if args.mode == "decay_sweep":
        print("\n=== Decay sweep: alpha error vs Hough decay ===\n")
        for decay in [0.999, 0.99, 0.95, 0.9]:
            run_falling_pencil_benchmark(
                n_trials=30,
                total_time=0.5,
                alpha_max=0.05,
                hough_decay=decay,
                show_progress=False,
            )
