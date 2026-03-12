"""
Realtime DVS camera calibration viewer.

Shows recent events from two DAVIS346 cameras with no line fitting or other
processing, so camera focus / focal setup can be adjusted against a live feed.

Usage:
    python3 hardware/dvs_camera_calibration.py
    python3 hardware/dvs_camera_calibration.py --cam1 SERIAL1 --cam2 SERIAL2
    python3 hardware/dvs_camera_calibration.py --display-fps 30 --decay-display 0.5
"""

import argparse
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perception.dvs_camera_reader import (
    DAVIS346_HEIGHT,
    DAVIS346_WIDTH,
    DVSReader,
    discover_devices,
)


class CameraEventPreview:
    """Continuously updates a recent-event surface for one camera."""

    def __init__(self, device_or_serial: str, decay_display: float, noise_filter_duration_ms: float | None = None):
        self.reader = DVSReader(device_or_serial, noise_filter_duration_ms=noise_filter_duration_ms)
        self.decay_display = decay_display
        self.surface = np.zeros((DAVIS346_HEIGHT, DAVIS346_WIDTH), dtype=np.float32)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        while not self._stop.is_set() and self.reader.is_running():
            events = self.reader.get_event_batch()
            if events is None or len(events) == 0:
                time.sleep(0.0005)
                continue

            xs = events["x"]
            ys = events["y"]
            with self._lock:
                self.surface *= self.decay_display
                np.add.at(self.surface, (ys, xs), 1.0)

    def get_surface(self) -> np.ndarray:
        with self._lock:
            return self.surface.copy()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.reader.close()


def build_display_frame(surface: np.ndarray, intensity_gain: float) -> np.ndarray:
    frame = np.clip(surface * intensity_gain, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show recent DVS camera events for focus / calibration.")
    parser.add_argument("--cam1", help="Camera 1 serial or device (omit to use discovery)")
    parser.add_argument("--cam2", help="Camera 2 serial or device (omit to use discovery)")
    parser.add_argument(
        "--display-fps",
        type=float,
        default=30.0,
        help="Target GUI refresh rate in frames per second.",
    )
    parser.add_argument(
        "--decay-display",
        type=float,
        default=0.5,
        help="Per-batch display decay. Lower values keep only more recent events.",
    )
    parser.add_argument(
        "--surface-intensity-gain",
        type=float,
        default=50.0,
        help="Scales surface brightness before clipping to 8-bit.",
    )
    parser.add_argument(
        "--noise-filter-duration",
        type=float,
        default=None,
        metavar="MS",
        help="Optional event noise filter duration in milliseconds. Omit for raw event preview.",
    )
    return parser.parse_args()


def resolve_devices(args: argparse.Namespace) -> tuple[str, str]:
    if args.cam1 is not None and args.cam2 is not None:
        print("Opening cameras (from serials)...")
        return args.cam1, args.cam2

    if args.cam1 is not None or args.cam2 is not None:
        print("Error: Provide both --cam1 and --cam2, or omit both to use discovery.", file=sys.stderr)
        sys.exit(1)

    devices = discover_devices()
    print(f"Discovered devices: {devices}")
    if len(devices) < 2:
        print("Error: Need at least 2 DVS cameras. Connect both or pass --cam1 and --cam2.", file=sys.stderr)
        sys.exit(1)

    print("Using devices[0] and devices[1] for x and y cams")
    return devices[0], devices[1]


def main() -> None:
    args = parse_args()
    device1, device2 = resolve_devices(args)

    display_period = 1.0 / args.display_fps
    next_display = time.perf_counter()

    preview1 = CameraEventPreview(
        device1,
        decay_display=args.decay_display,
        noise_filter_duration_ms=args.noise_filter_duration,
    )
    preview2 = CameraEventPreview(
        device2,
        decay_display=args.decay_display,
        noise_filter_duration_ms=args.noise_filter_duration,
    )

    cv2.namedWindow("Cam 1 Preview", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Cam 2 Preview", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Cam 1 Preview", 50, 100)
    cv2.moveWindow("Cam 2 Preview", 50 + DAVIS346_WIDTH + 55, 100)

    print("Running calibration preview. Press 'q' to quit.")

    try:
        while preview1.reader.is_running() and preview2.reader.is_running():
            now = time.perf_counter()
            if now < next_display:
                time.sleep(min(0.001, next_display - now))
                continue

            frame1 = build_display_frame(preview1.get_surface(), args.surface_intensity_gain)
            frame2 = build_display_frame(preview2.get_surface(), args.surface_intensity_gain)

            cv2.imshow("Cam 1 Preview", frame1)
            cv2.imshow("Cam 2 Preview", frame2)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            while next_display <= now:
                next_display += display_period
    finally:
        preview1.close()
        preview2.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
