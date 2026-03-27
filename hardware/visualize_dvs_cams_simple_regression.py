"""
Standalone DVS camera visualization with line overlay + simple regression pose preview.

Same as `hardware/visualize_dvs_cams.py` for camera IO, masking, and line tracking,
but the window title shows pose estimates (X, Y, alpha_x, alpha_y) from
`SimpleDVSRegressionModel` (affine v1 or `dvs_calibration_dataset.json`). Pixel
line fits are converted to camnorm with `CameraModel.pixel_to_camnorm` before
`estimate_pose`, matching the model API.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from core.sim_types import CameraPair, HoughTrackerParams
from visualization.composite_layout import build_composite, get_default_window_size
from perception.dvs_camera_reader import DVSReader, discover_devices, DAVIS346_WIDTH, DAVIS346_HEIGHT
from perception.dvs_algorithms import PaperHoughLineAlgorithm, SamLineAlgorithm, mask_events_below_line, line_x_at_pixel_y
from perception.camera_model import CameraModel
from perception.simple_dvs_regression_model import SimpleDVSRegressionModel


def _window_closed(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize DVS cams and show pose from simple regression model")
    parser.add_argument("--cam1", help="Camera 1 serial or device (omit to use discovery)")
    parser.add_argument("--cam2", help="Camera 2 serial or device (omit to use discovery)")
    parser.add_argument(
        "--mode",
        choices=["hough", "sam"],
        default="hough",
        help="Line algorithm: hough (paper tracker) or sam (OLS on events)",
    )
    parser.add_argument("--noise-filter-duration", type=float, default=None, metavar="MS", help="Noise filter (ms)")
    parser.add_argument("--mask-y-cam1", type=int, default=160, metavar="Y", help="ROI mask y for cam1")
    parser.add_argument("--mask-y-cam2", type=int, default=190, metavar="Y", help="ROI mask y for cam2")
    parser.add_argument("--decay-display", type=float, default=0.5, help="Event surface decay")
    parser.add_argument("--surface-intensity-gain", type=float, default=50.0, help="Surface brightness")
    parser.add_argument("--display-fps", type=float, default=30.0, help="GUI refresh rate")
    parser.add_argument(
        "--max-events-per-batch",
        type=int,
        default=None,
        metavar="N",
        help="Hough only: cap events fed to the algorithm per iteration (newest N kept).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hardware/calibration_files/dvs_calibration_dataset.json",
        help="Affine v1 JSON (simple_dvs_regression_v1) or combined b1/b2/s1/s2 dataset JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = SimpleDVSRegressionModel.load(Path(args.model))

    if args.cam1 is not None and args.cam2 is not None:
        device1, device2 = args.cam1, args.cam2
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

    reader1 = DVSReader(device1, noise_filter_duration_ms=args.noise_filter_duration)
    reader2 = DVSReader(device2, noise_filter_duration_ms=args.noise_filter_duration)

    if args.mode == "hough":
        hough_params = HoughTrackerParams()
        algo1 = PaperHoughLineAlgorithm(width=DAVIS346_WIDTH, height=DAVIS346_HEIGHT, params=hough_params, max_events=args.max_events_per_batch)
        algo2 = PaperHoughLineAlgorithm(width=DAVIS346_WIDTH, height=DAVIS346_HEIGHT, params=hough_params, max_events=args.max_events_per_batch)
    else:
        algo1 = SamLineAlgorithm(width=DAVIS346_WIDTH, height=DAVIS346_HEIGHT, min_points=50)
        algo2 = SamLineAlgorithm(width=DAVIS346_WIDTH, height=DAVIS346_HEIGHT, min_points=50)

    W, H = DAVIS346_WIDTH, DAVIS346_HEIGHT
    cam_model = CameraModel(width=W, height=H)
    decay_display = float(args.decay_display)
    surface_intensity_gain = float(args.surface_intensity_gain)
    display_period = 1.0 / float(args.display_fps)
    next_display = time.perf_counter()
    surface1 = np.zeros((H, W), dtype=np.float32)
    surface2 = np.zeros((H, W), dtype=np.float32)

    WINDOW_NAME = "DVS cam preview (simple regression pose)"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    w, h = get_default_window_size(has_cams=True, has_workspace=False)
    cv2.resizeWindow(WINDOW_NAME, w, h)

    print("Running. Press 'q' to quit.")
    result1, result2 = None, None
    while reader1.is_running() and reader2.is_running():
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
        if now < next_display:
            continue

        frame1 = np.clip(surface1 * surface_intensity_gain, 0, 255).astype(np.uint8)
        frame2 = np.clip(surface2 * surface_intensity_gain, 0, 255).astype(np.uint8)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)

        # Draw overlays + compute pose if available.
        pose_str = "pose=?"
        if result1 is not None and not isinstance(result1, tuple) and result2 is not None and not isinstance(result2, tuple):
            cams = CameraPair(
                cam1=cam_model.pixel_to_camnorm(result1),
                cam2=cam_model.pixel_to_camnorm(result2),
            )
            pose = model.estimate_pose(cams, cam_model)
            pose_str = f"X={pose.X:+.4f} Y={pose.Y:+.4f} ax={pose.alpha_x:+.3f} ay={pose.alpha_y:+.3f}"

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

        title = f"{args.mode} | {pose_str} | Q: quit"
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


if __name__ == "__main__":
    main()

