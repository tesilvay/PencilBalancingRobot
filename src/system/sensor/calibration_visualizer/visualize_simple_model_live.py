from __future__ import annotations

import argparse
import sys
import time

from src.experiment.realtime_visualizer.real_dvs_workspace import (
    RealDvsWorkspaceVisualizer,
    RealDvsWorkspaceVisualizerParams,
)
from src.shared import (
    ControlInput,
    State,
    build_from_registry,
    default_camera_params,
    default_workspace,
    resolve_preset,
)
from src.system.sensor.algo import LINE_ALGO_REGISTRY
from src.system.sensor.observation_model import REG_MODEL_REGISTRY
from src.system.sensor.interface.real_dvs import REAL_DVS_PRESETS, RealDVSParams, RealEventCameraInterface
from src.system.sensor.observation_model.simple_dvs import (
    SimpleDVSRegressionModelLoader,
    SimpleDVSRegressionModelParams,
)
from src.system.sensor.observation_model.simple_dvs_regression_model import default_affine_calibration_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the real DVS sensor like the real system and preview it with the "
            "real DVS workspace visualizer."
        )
    )
    parser.add_argument("--cam1", help="Camera 1 serial or device path")
    parser.add_argument("--cam2", help="Camera 2 serial or device path")
    parser.add_argument(
        "--sensor",
        default="real_dvs:hough",
        help="Real sensor spec to mirror the real system, for example real_dvs:hough or real_dvs:sam",
    )
    parser.add_argument(
        "--algo",
        default=None,
        help="Optional line algorithm override, for example hough:default or sam:default",
    )
    parser.add_argument(
        "--obs-model",
        default=None,
        help="Optional regression/observation model override, for example simple:default or analytic:default",
    )
    parser.add_argument(
        "--model",
        default=str(default_affine_calibration_path()),
        help="Path used when --obs-model is omitted or set to simple:default",
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
        help="Clamp used by the simple regression model loader when --obs-model is simple/default",
    )
    return parser.parse_args()


def _title_from_measurement(y_meas) -> str:
    if y_meas is None:
        return "Real DVS workspace preview | waiting for sensor lock | Q: quit"
    return "Real DVS workspace preview | command=(0, 0) | Q: quit"


def _zero_state() -> State:
    return State(
        px=0.0,
        vx=0.0,
        ax=0.0,
        wx=0.0,
        py=0.0,
        vy=0.0,
        ay=0.0,
        wy=0.0,
    )


def _build_default_simple_model(args: argparse.Namespace):
    return SimpleDVSRegressionModelLoader(
        SimpleDVSRegressionModelParams(
            calibration_path=args.model,
            max_tilt_deg=args.max_tilt_deg,
        )
    )


def _build_sensor(args: argparse.Namespace) -> RealEventCameraInterface:
    sensor_type, sensor_preset = args.sensor.split(":")
    if sensor_type != "real_dvs":
        raise ValueError(f"This preview only supports real_dvs sensors, got {args.sensor!r}")

    preset = resolve_preset(REAL_DVS_PRESETS, sensor_preset)
    cam_params = default_camera_params()
    algo = build_from_registry(LINE_ALGO_REGISTRY, preset["algo"])
    obs_model = build_from_registry(REG_MODEL_REGISTRY, preset["obs_model"])
    noise_filter_duration_ms = preset.get("noise_filter_duration_ms")

    if args.algo is not None:
        algo = build_from_registry(LINE_ALGO_REGISTRY, args.algo)
    if args.obs_model is not None:
        obs_model = build_from_registry(REG_MODEL_REGISTRY, args.obs_model)
    elif str(preset["obs_model"]).startswith("simple:"):
        obs_model = _build_default_simple_model(args)
    if args.noise_filter_duration is not None:
        noise_filter_duration_ms = args.noise_filter_duration

    sensor = RealEventCameraInterface(
        RealDVSParams(
            algo=algo,
            obs_model=obs_model,
            cam_params=cam_params,
            cam1_device=args.cam1,
            cam2_device=args.cam2,
            noise_filter_duration_ms=noise_filter_duration_ms,
        )
    )
    return sensor


def main() -> int:
    args = parse_args()

    if (args.cam1 is None) != (args.cam2 is None):
        print("Provide both --cam1 and --cam2, or omit both to auto-discover.", file=sys.stderr)
        return 1

    sensor = _build_sensor(args)
    visualizer = RealDvsWorkspaceVisualizer(
        RealDvsWorkspaceVisualizerParams(
            width=int(sensor.cam_width_px),
            height=int(sensor.cam_height_px),
            top_mask_y_cam1=int(sensor._dvs_top_mask_line_y_cam1),
            top_mask_y_cam2=int(sensor._dvs_top_mask_line_y_cam2),
            mask_y_cam1=int(sensor._dvs_mask_line_y_cam1),
            mask_y_cam2=int(sensor._dvs_mask_line_y_cam2),
            workspace=default_workspace(),
            event_frames_fn=sensor.get_event_accumulator_frames,
        )
    )

    display_period = 1.0 / max(float(args.display_fps), 1e-6)
    next_display = time.perf_counter()
    zero_state = _zero_state()
    null_command = ControlInput(px_cmd=0.0, py_cmd=0.0)

    print("Running real DVS workspace preview with command=(0, 0). Press Q or Esc to quit.")
    try:
        while True:
            now = time.perf_counter()
            if now < next_display:
                time.sleep(min(0.001, next_display - now))
                continue

            y_meas = None
            try:
                y_meas = sensor.get_y(zero_state)
            except RuntimeError:
                pass

            vr = visualizer.render(
                measurement=getattr(sensor, "last_line_observation", None),
                command=null_command,
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
