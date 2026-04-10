from __future__ import annotations

import argparse
import sys
import time

from src.experiment.realtime_visualizer.real_dvs import (
    RealDvsVisualizer,
    RealDvsVisualizerParams,
)
from src.shared import CameraPair, default_camera_params
from src.shared import build_from_registry
from src.system.sensor.algo import LINE_ALGO_REGISTRY
from src.system.sensor.interface.real_dvs import RealDVSParams, RealEventCameraInterface
from src.system.sensor.observation_model.simple_dvs import (
    SimpleDVSRegressionModelLoader,
    SimpleDVSRegressionModelParams,
)
from src.system.sensor.observation_model.simple_dvs_regression_model import (
    SimpleDVSRegressionModel,
    default_affine_calibration_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the real DVS cameras and preview the simple regression calibration "
            "using the existing real visualizer."
        )
    )
    parser.add_argument("--cam1", help="Camera 1 serial or device path")
    parser.add_argument("--cam2", help="Camera 2 serial or device path")
    parser.add_argument(
        "--mode",
        choices=["hough", "sam"],
        default="hough",
        help="Line tracking algorithm used by the real camera interface",
    )
    parser.add_argument(
        "--model",
        default=str(default_affine_calibration_path()),
        help="Path to the simple DVS regression calibration JSON",
    )
    parser.add_argument(
        "--noise-filter-duration",
        type=float,
        default=None,
        metavar="MS",
        help="Optional camera noise filter duration in milliseconds",
    )
    parser.add_argument(
        "--display-fps",
        type=float,
        default=30.0,
        help="GUI refresh rate",
    )
    parser.add_argument(
        "--max-tilt-deg",
        type=float,
        default=30.0,
        help="Clamp used by the simple regression model loader",
    )
    return parser.parse_args()


def _title_from_measurement(y_meas) -> str:
    if y_meas is None:
        return "Simple model y_meas preview | waiting for both cameras | Q: quit"
    return "y_meas preview"


def _build_sensor(args: argparse.Namespace) -> RealEventCameraInterface:
    simple_model = SimpleDVSRegressionModel.load(
        args.model,
        max_tilt_deg=args.max_tilt_deg,
    )
    cam_params = default_camera_params()
    cam_params.y_mask_line_1 = int(simple_model.mask_y_cam1)
    cam_params.y_mask_line_2 = int(simple_model.mask_y_cam2)

    algo = build_from_registry(LINE_ALGO_REGISTRY, f"{args.mode}:default")
    obs_model = SimpleDVSRegressionModelLoader(
        SimpleDVSRegressionModelParams(
            calibration_path=args.model,
            max_tilt_deg=args.max_tilt_deg,
        )
    )

    sensor = RealEventCameraInterface(
        RealDVSParams(
            algo=algo,
            obs_model=obs_model,
            cam_params=cam_params,
            cam1_device=args.cam1,
            cam2_device=args.cam2,
            noise_filter_duration_ms=args.noise_filter_duration,
        )
    )
    return sensor


def main() -> int:
    args = parse_args()

    if (args.cam1 is None) != (args.cam2 is None):
        print("Provide both --cam1 and --cam2, or omit both to auto-discover.", file=sys.stderr)
        return 1

    sensor = _build_sensor(args)
    visualizer = RealDvsVisualizer(
        RealDvsVisualizerParams(
            width=int(sensor.cam_width_px),
            height=int(sensor.cam_height_px),
            mask_y_cam1=int(sensor._dvs_mask_line_y_cam1),
            mask_y_cam2=int(sensor._dvs_mask_line_y_cam2),
            event_frames_fn=sensor.get_event_accumulator_frames,
        )
    )

    display_period = 1.0 / max(float(args.display_fps), 1e-6)
    next_display = time.perf_counter()

    print("Running simple calibration preview. Press Q or Esc to quit.")
    try:
        while True:
            now = time.perf_counter()
            if now < next_display:
                time.sleep(min(0.001, next_display - now))
                continue

            measurement: CameraPair | None = sensor.get_z()
            y_meas = None
            if measurement is not None:
                cams_px = CameraPair(
                    cam1=sensor.cam.camnorm_to_pixel(measurement.cam1),
                    cam2=sensor.cam.camnorm_to_pixel(measurement.cam2),
                )
                y_meas = sensor.dvs_regression_model.estimate(cams_px)

            vr = visualizer.render(
                measurement=measurement,
                title=_title_from_measurement(y_meas),
                y_meas=y_meas,
            )
            if vr.quit:
                break

            while next_display <= now:
                next_display += display_period
    finally:
        sensor.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
