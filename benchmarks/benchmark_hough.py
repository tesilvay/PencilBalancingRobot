import numpy as np
import matplotlib.pyplot as plt

from perception.dvs_algorithms import PaperHoughLineAlgorithm
from perception.camera_model import CameraModel
from core.sim_types import CameraObservation


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
# Single trial
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

        if obs_px.slope is None:
            continue

        # convert pixel line → normalized line
        obs_est = cam.pixel_to_normalized(obs_px)

        b_est_hist.append(obs_est.intercept)
        s_est_hist.append(obs_est.slope)

    return np.array(b_est_hist), np.array(s_est_hist)


# -------------------------------------------------
# Monte Carlo test
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



if __name__ == "__main__":

    run_hough_monte_carlo(
        n_trials=20,
        noise_px=1.0
    )

