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
"""

import argparse
import sys
import cv2
import numpy as np

from perception.dvs_camera_reader import DVSReader, discover_devices, DAVIS346_WIDTH, DAVIS346_HEIGHT
from perception.dvs_algorithms import PaperHoughLineAlgorithm, SamLineAlgorithm


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
    parser.add_argument("--decay", type=float, default=0.95, help="Hough decay, only for --mode hough (default 0.95)")
    parser.add_argument(
        "--noise-filter-duration",
        type=float,
        default=None,
        metavar="MS",
        help="Noise filter duration (ms). Omit = no filter. Use 30 to match main.py with Sam.",
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
        algo1 = PaperHoughLineAlgorithm(width=DAVIS346_WIDTH, height=DAVIS346_HEIGHT, decay=args.decay)
        algo2 = PaperHoughLineAlgorithm(width=DAVIS346_WIDTH, height=DAVIS346_HEIGHT, decay=args.decay)
    else:
        algo1 = SamLineAlgorithm(width=DAVIS346_WIDTH, height=DAVIS346_HEIGHT, min_points=50)
        algo2 = SamLineAlgorithm(width=DAVIS346_WIDTH, height=DAVIS346_HEIGHT, min_points=50)

    print(f"Using {args.mode} line algorithm.")
    # cam_model = CameraModel(width=DAVIS346_WIDTH, height=DAVIS346_HEIGHT)

    W, H = DAVIS346_WIDTH, DAVIS346_HEIGHT
    decay_display = 0.95
    surface1 = np.zeros((H, W), dtype=np.float32)
    surface2 = np.zeros((H, W), dtype=np.float32)

    cv2.namedWindow("Cam 1 (x-axis)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Cam 2 (y-axis)", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Cam 1 (x-axis)", 50, 100)
    cv2.moveWindow("Cam 2 (y-axis)", 50 + W + 55, 100)

    print("Running. Press 'q' to quit.")

    result1, result2 = None, None
    while reader1.is_running() and reader2.is_running():
        events1 = reader1.get_event_batch()
        events2 = reader2.get_event_batch()

        if events1 is not None and len(events1) > 0:
            surface1 *= decay_display
            xs, ys = events1["x"], events1["y"]
            np.add.at(surface1, (ys, xs), 1.0)
            result1 = algo1.update(events1)

        if events2 is not None and len(events2) > 0:
            surface2 *= decay_display
            xs, ys = events2["x"], events2["y"]
            np.add.at(surface2, (ys, xs), 1.0)
            result2 = algo2.update(events2)

        # Build display frames (normalize surface to 0-255)
        frame1 = np.clip(surface1 * 50, 0, 255).astype(np.uint8)
        frame2 = np.clip(surface2 * 50, 0, 255).astype(np.uint8)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)

        # Draw detected line on each frame
        for frame, result in [(frame1, result1), (frame2, result2)]:
            if result is not None and not isinstance(result, tuple):
                obs_px = result
                s_px, b_px = obs_px.slope, obs_px.intercept
                y0, y1 = 0, H - 1
                x0 = int(s_px * y0 + b_px)
                x1 = int(s_px * y1 + b_px)
                cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

        cv2.imshow("Cam 1 (x-axis)", frame1)
        cv2.imshow("Cam 2 (y-axis)", frame2)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    reader1.close()
    reader2.close()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
