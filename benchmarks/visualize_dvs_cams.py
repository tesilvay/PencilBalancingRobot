"""
Standalone DVS camera visualization with Hough line overlay.

Initializes both DAVIS346 cameras, runs Hough in the main loop,
and renders accumulated events with the detected line overlaid.
Use this to verify cams and algo before plugging into main.py (HIL).

Usage:
    python -m benchmarks.visualize_dvs_cams --cam1 SERIAL1 --cam2 SERIAL2

Example (use serials from dv-list-devices):
    python -m benchmarks.visualize_dvs_cams --cam1 00000499 --cam2 00000500
"""

import argparse
import cv2
import numpy as np

from perception.dvs_camera_reader import DVSReader, DAVIS346_WIDTH, DAVIS346_HEIGHT
from perception.dvs_algorithms import PaperHoughLineAlgorithm


def main():
    parser = argparse.ArgumentParser(description="Visualize DVS cams with Hough line overlay")
    parser.add_argument("--cam1", required=True, help="Camera 1 serial (e.g. from dv-list-devices)")
    parser.add_argument("--cam2", required=True, help="Camera 2 serial")
    parser.add_argument("--decay", type=float, default=0.95, help="Hough decay (default 0.95)")
    args = parser.parse_args()

    print("Opening cameras...")
    reader1 = DVSReader(args.cam1)
    reader2 = DVSReader(args.cam2)

    algo1 = PaperHoughLineAlgorithm(width=DAVIS346_WIDTH, height=DAVIS346_HEIGHT, decay=args.decay)
    algo2 = PaperHoughLineAlgorithm(width=DAVIS346_WIDTH, height=DAVIS346_HEIGHT, decay=args.decay)
    cam_model = CameraModel(width=DAVIS346_WIDTH, height=DAVIS346_HEIGHT)

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

        # Draw Hough line on each frame
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
