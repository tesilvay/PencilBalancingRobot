"""
Standalone DVS camera visualization with line overlay.

Initializes both DAVIS346 cameras, runs a line-tracking algorithm (Hough or Sam),
and renders accumulated events with the detected line overlaid.
Use this to verify cams and algo before plugging into main.py (HIL).

Usage:
    # Auto-discover, default Hough algorithm:
    python -m benchmarks.visualize_dvs_cams

    # Sam's OLS algorithm:
    python -m benchmarks.visualize_dvs_cams --mode sam

    # Match main.py pipeline (Sam + noise filter 30 ms):
    python -m benchmarks.visualize_dvs_cams --mode sam --noise-filter-duration 30

    # Or specify serials explicitly:
    python -m benchmarks.visualize_dvs_cams --cam1 SERIAL1 --cam2 SERIAL2
    
    sudo .venv/bin/python -m hardware.visualize_dvs_cams
"""

import argparse
import sys
import time
import cv2
import numpy as np

from core.sim_types import HoughTrackerParams
from visualization.composite_layout import build_composite, get_default_window_size
from perception.dvs_camera_reader import DVSReader, discover_devices, DAVIS346_WIDTH, DAVIS346_HEIGHT
from perception.dvs_algorithms import PaperHoughLineAlgorithm, SamLineAlgorithm, mask_events_below_line, line_x_at_pixel_y


def _window_closed(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def main():
    parser = argparse.ArgumentParser(description="Visualize DVS cams with Hough line overlay")
    parser.add_argument("--cam1", help="Camera 1 serial or device (omit to use discovery)")
    parser.add_argument("--cam2", help="Camera 2 serial or device (omit to use discovery)")
    parser.add_argument(
        "--mode",
        choices=["hough", "sam"],
        default="hough",
        help="Line algorithm: hough (paper tracker) or sam (OLS on events)",
    )
    parser.add_argument(
        "--hough-mixing-factor",
        type=float,
        default=0.02,
        help="Hough only: per-event adaptation rate; 0.01-0.05 is a good starting range, larger is faster and noisier.",
    )
    parser.add_argument(
        "--hough-inlier-stddev-px",
        type=float,
        default=4.0,
        help="Hough only: Gaussian inlier width in pixels; 3-6 px is typical, larger follows faster motion but admits more background noise.",
    )
    parser.add_argument(
        "--hough-min-determinant",
        type=float,
        default=1e-6,
        help="Hough only: reject unstable solves when the quadratic is near-singular; usually leave near 1e-6.",
    )
    parser.add_argument(
        "--noise-filter-duration",
        type=float,
        default=None,
        metavar="MS",
        help="Noise filter duration (ms). Omit = no filter. Use 30 to match main.py with Sam.",
    )
    parser.add_argument(
        "--mask-y-cam1",
        type=int,
        default=160,
        metavar="Y",
        help="ROI mask line y for cam1. Events with y >= Y are ignored (keeps y < Y).",
    )
    parser.add_argument(
        "--mask-y-cam2",
        type=int,
        default=190,
        metavar="Y",
        help="ROI mask line y for cam2. Events with y >= Y are ignored (keeps y < Y).",
    )
    parser.add_argument(
        "--decay-display",
        type=float,
        default=0.5,
        help="Display only: per-batch surface decay. Lower values keep only more recent events.",
    )
    parser.add_argument(
        "--surface-intensity-gain",
        type=float,
        default=50.0,
        help="Display only: scales surface brightness before clipping to 8-bit.",
    )
    parser.add_argument(
        "--display-fps",
        type=float,
        default=30.0,
        help="Display only: target GUI refresh rate in frames per second.",
    )
    parser.add_argument(
        "--max-events-per-batch",
        type=int,
        default=None,
        metavar="N",
        help="Hough only: cap events fed to the algorithm per iteration (newest N kept). "
             "Omit to process all events (Numba JIT handles typical rates).",
    )
    args = parser.parse_args()

    if args.cam1 is not None and args.cam2 is not None:
        device1, device2 = args.cam1, args.cam2
        print("Opening cameras (from serials)...")
    elif args.cam1 is not None or args.cam2 is not None:
        print("Error: Provide both --cam1 and --cam2, or omit both to use discovery.", file=sys.stderr)
        sys.exit(1)
    else:
        devices = discover_devices()
        print(f"Discovered devices: {devices}")
        if len(devices) < 2:
            print("Error: Need at least 2 DVS cameras. Connect both or pass --cam1 and --cam2.", file=sys.stderr)
            sys.exit(1)
        device1, device2 = devices[0], devices[1]
        print(f"Using devices[0] and devices[1] for x and y cams")

    reader1 = DVSReader(device1, noise_filter_duration_ms=args.noise_filter_duration)
    reader2 = DVSReader(device2, noise_filter_duration_ms=args.noise_filter_duration)

    if args.mode == "hough":
        hough_params = HoughTrackerParams(
            mixing_factor=args.hough_mixing_factor,
            inlier_stddev_px=args.hough_inlier_stddev_px,
            min_determinant=args.hough_min_determinant,
        )
        algo1 = PaperHoughLineAlgorithm(
            width=DAVIS346_WIDTH, height=DAVIS346_HEIGHT,
            params=hough_params, max_events=args.max_events_per_batch,
        )
        algo2 = PaperHoughLineAlgorithm(
            width=DAVIS346_WIDTH, height=DAVIS346_HEIGHT,
            params=hough_params, max_events=args.max_events_per_batch,
        )
    else:
        algo1 = SamLineAlgorithm(width=DAVIS346_WIDTH, height=DAVIS346_HEIGHT, min_points=50)
        algo2 = SamLineAlgorithm(width=DAVIS346_WIDTH, height=DAVIS346_HEIGHT, min_points=50)

    print(f"Using {args.mode} line algorithm.")
    # cam_model = CameraModel(width=DAVIS346_WIDTH, height=DAVIS346_HEIGHT)

    W, H = DAVIS346_WIDTH, DAVIS346_HEIGHT
    decay_display = args.decay_display
    surface_intensity_gain = args.surface_intensity_gain
    display_period = 1.0 / args.display_fps
    next_display = time.perf_counter()
    surface1 = np.zeros((H, W), dtype=np.float32)
    surface2 = np.zeros((H, W), dtype=np.float32)

    WINDOW_NAME = "DVS cam preview"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    w, h = get_default_window_size(has_cams=True, has_workspace=False)
    cv2.resizeWindow(WINDOW_NAME, w, h)

    print("Running. Press 'q' to quit.")

    result1, result2 = None, None
    while reader1.is_running() and reader2.is_running():
        # Drain all queued batches per camera to prevent backlog.
        # Surface gets every event (visual accuracy); algorithm sees the
        # merged batch (event cap inside the algo trims if needed).
        batches1 = []
        while True:
            b = reader1.get_event_batch()
            if b is None or len(b) == 0:
                break
            batches1.append(b)

        batches2 = []
        while True:
            b = reader2.get_event_batch()
            if b is None or len(b) == 0:
                break
            batches2.append(b)

        if batches1:
            events1 = np.concatenate(batches1)
            events1 = mask_events_below_line(events1, mask_line_y=args.mask_y_cam1, frame_height=H)
            surface1 *= decay_display
            if len(events1) > 0:
                np.add.at(surface1, (events1["y"], events1["x"]), 1.0)
            result1 = algo1.update(events1)

        if batches2:
            events2 = np.concatenate(batches2)
            events2 = mask_events_below_line(events2, mask_line_y=args.mask_y_cam2, frame_height=H)
            surface2 *= decay_display
            if len(events2) > 0:
                np.add.at(surface2, (events2["y"], events2["x"]), 1.0)
            result2 = algo2.update(events2)

        if not batches1 and not batches2:
            time.sleep(0.0001)

        now = time.perf_counter()
        if now >= next_display:
            # Build display frames (normalize surface to 0-255).
            frame1 = np.clip(surface1 * surface_intensity_gain, 0, 255).astype(np.uint8)
            frame2 = np.clip(surface2 * surface_intensity_gain, 0, 255).astype(np.uint8)
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)

            # Draw detected line on each frame.
            for frame, result, mask_y in [(frame1, result1, args.mask_y_cam1), (frame2, result2, args.mask_y_cam2)]:
                if 0 < mask_y < H:
                    cv2.line(frame, (0, mask_y), (W - 1, mask_y), (0, 165, 255), 2)
                if result is not None and not isinstance(result, tuple):
                    obs_px = result
                    s_px, b_px = obs_px.slope, obs_px.intercept
                    y0 = 0
                    y1 = (min(mask_y - 1, H - 1) if 0 < mask_y < H else (H - 1))
                    x0 = int(round(s_px * y0 + b_px))
                    x1 = int(round(s_px * y1 + b_px))
                    cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    if 0 < mask_y < H:
                        xi = int(round(line_x_at_pixel_y(obs_px, mask_y)))
                        if 0 <= xi < W:
                            cv2.circle(frame, (xi, mask_y), 5, (0, 255, 0), -1)

            def _fmt_s_xatmask(res, mask_y: int) -> str:
                if res is None or isinstance(res, tuple):
                    return "s=?, x_at_mask=?"
                try:
                    s = float(res.slope)
                    x_at_mask = float(line_x_at_pixel_y(res, mask_y))
                except (TypeError, ValueError):
                    return "s=?, x_at_mask=?"
                if not np.isfinite(s) or not np.isfinite(x_at_mask):
                    return "s=?, x_at_mask=?"
                return f"s={s:+.4f}, x_at_mask={x_at_mask:+.1f}"

            title = (
                f"{args.mode} | "
                f"cam1({_fmt_s_xatmask(result1, args.mask_y_cam1)}) "
                f"cam2({_fmt_s_xatmask(result2, args.mask_y_cam2)}) | Q: quit"
            )
            composite = build_composite(title, frame1, frame2, None)
            cv2.imshow(WINDOW_NAME, composite)

            if _window_closed(WINDOW_NAME):
                break
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break

            while next_display <= now:
                next_display += display_period

    reader1.close()
    reader2.close()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
