"""
Interactive simple DVS regression calibrator.

Default workflow: **b1** → **b2** → **s1** → **s2** (see ``hardware/dvs_bx_calibration.py`` and
``hardware/dvs_sx_calibration.py``), then one combined JSON dataset.

B1/B2: A/D moves lateral line position (``x_at_mask``); slope fixed.

S1/S2: A/D adjusts slope only; line kept at image center (``x_at_mask`` pinned); b changes only
as required by geometry for that slope.

Optional ``--full-regression``: legacy four-stage affine fit (both cameras) and save
``simple_dvs_regression.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.sim_types import MechanismParams, PhysicalParams, PlantParams, RunParams, TableCommand, WorkspaceParams
from perception.camera_model import CameraModel
from perception.dvs_algorithms import mask_events_below_line
from perception.dvs_camera_reader import DAVIS346_HEIGHT, DAVIS346_WIDTH, DVSReader, discover_devices
from perception.simple_dvs_regression_model import SimpleDVSCameraCalibration, SimpleDVSRegressionModel, fit_affine
from visualization.composite_layout import build_composite, get_default_window_size

from hardware.dvs_calibration.dvs_bx_calibration import (
    B1InterceptCalibration,
    B1Sample,
    B2InterceptCalibration,
    B2Sample,
    ManualLineState,
    x_positions_from_safe_radius,
)
from hardware.dvs_calibration.dvs_sx_calibration import (
    DEFAULT_TILT_CALIB_DEGS,
    S1Sample,
    S1SlopeCalibration,
    S2Sample,
    S2SlopeCalibration,
    tilt_degs_to_rads,
)
from visualization.realtime_visualizer import OneDvsVisualizer


def _window_closed(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


DEFAULT_WORKSPACE = WorkspaceParams(x_ref=0.0, y_ref=0.0, safe_radius=0.068)
DEFAULT_MECHANISM = MechanismParams(
    O=(128.77, 178.13),
    B=(101.77, 210.13),
    la=175,
    lb=175,
)


def draw_manual_overlay(frame: np.ndarray, state: ManualLineState, mask_y: int) -> None:
    H, W = frame.shape[:2]
    if 0 < mask_y < H:
        cv2.line(frame, (0, mask_y), (W - 1, mask_y), (0, 165, 255), 2)
    s_px, b_px = state.to_obs_px(mask_y=mask_y)
    y0 = 0
    y1 = min(mask_y - 1, H - 1) if 0 < mask_y < H else (H - 1)
    x0 = int(round(s_px * y0 + b_px))
    x1 = int(round(s_px * y1 + b_px))
    x0 = max(-10_000, min(10_000, x0))
    x1 = max(-10_000, min(10_000, x1))
    try:
        cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
    except cv2.error:
        return
    if 0 < mask_y < H:
        xi = int(round(state.x_at_mask_px))
        if 0 <= xi < W:
            cv2.circle(frame, (xi, mask_y), 5, (0, 255, 0), -1)


def save_calibration_dataset_json(
    path: Path,
    *,
    b1_samples: list[B1Sample],
    b2_samples: list[B2Sample],
    s1_samples: list[S1Sample],
    s2_samples: list[S2Sample],
    x_targets_m: list[float],
    tilt_targets_deg: list[float],
    device_id_cam1: str,
    device_id_cam2: str,
    mask_y_cam1: int,
    mask_y_cam2: int,
    safe_radius_m: float,
    x_step_m: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "stages": ["b1", "b2", "s1", "s2"],
        "safe_radius_m": float(safe_radius_m),
        "x_step_m": float(x_step_m),
        "position_targets_m": [float(x) for x in x_targets_m],
        "tilt_targets_deg": [float(t) for t in tilt_targets_deg],
        "b1": {
            "stage": "b1",
            "camera_id": device_id_cam1,
            "mask_y_cam1": int(mask_y_cam1),
            "samples": [asdict(s) for s in b1_samples],
        },
        "b2": {
            "stage": "b2",
            "camera_id": device_id_cam2,
            "mask_y_cam2": int(mask_y_cam2),
            "samples": [asdict(s) for s in b2_samples],
        },
        "s1": {
            "stage": "s1",
            "camera_id": device_id_cam1,
            "mask_y_cam1": int(mask_y_cam1),
            "samples": [asdict(s) for s in s1_samples],
        },
        "s2": {
            "stage": "s2",
            "camera_id": device_id_cam2,
            "mask_y_cam2": int(mask_y_cam2),
            "samples": [asdict(s) for s in s2_samples],
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive simple DVS regression calibrator")
    p.add_argument(
        "--full-regression",
        action="store_true",
        help="Legacy: fit affine model for both cameras and save simple_dvs_regression.json",
    )
    p.add_argument("--cam1", help="Camera 1 serial/device (omit for discovery)")
    p.add_argument("--cam2", help="Camera 2 serial/device (omit for discovery)")
    p.add_argument("--noise-filter-duration", type=float, default=30.0, metavar="MS", help="Noise filter ms")
    p.add_argument("--mask-y-cam1", type=int, default=160, metavar="Y", help="Mask line y for cam1")
    p.add_argument("--mask-y-cam2", type=int, default=190, metavar="Y", help="Mask line y for cam2")
    p.add_argument("--decay-display", type=float, default=0.5, help="Event surface decay")
    p.add_argument("--surface-intensity-gain", type=float, default=50.0, help="Surface brightness (matches OneDvsVisualizer)")
    p.add_argument("--display-fps", type=float, default=30.0, help="GUI refresh rate")
    p.add_argument("--workspace-radius", type=float, default=DEFAULT_WORKSPACE.safe_radius, help="Workspace radius (m)")
    p.add_argument("--x-step-m", type=float, default=0.01, help="B1/B2 grid step along axis (m)")
    p.add_argument("--port", type=str, default="none", help="Servo port or 'none' for manual move")
    p.add_argument("--settle", type=float, default=2.0, metavar="SEC", help="Settle time after move (seconds)")
    p.add_argument("--tilt-deg-min", type=float, default=-10.0)
    p.add_argument("--tilt-deg-max", type=float, default=10.0)
    p.add_argument("--tilt-deg-step", type=float, default=5.0)
    p.add_argument("--n-position-points", type=int, default=10)
    p.add_argument(
        "--output",
        type=str,
        default="perception/calibration_files/simple_dvs_regression.json",
        help="Output JSON model path (full regression only)",
    )
    p.add_argument(
        "--dataset-output",
        type=str,
        default="hardware/calibration_files/dvs_calibration_dataset.json",
        help="Combined b1/b2/s1/s2 dataset JSON path",
    )
    p.add_argument("--step-x", type=float, default=1.0, help="B1/B2: A/D step for x_at_mask (pixels)")
    p.add_argument(
        "--step-s",
        type=float,
        default=0.005,
        help="S1/S2: A/D step for slope (px/px); legacy full regression uses W/S with same step",
    )
    return p.parse_args()


def _build_optional_actuator(port: str):
    port_norm = port.strip().lower()
    if port_norm == "none":
        return None
    # Lazy imports to avoid hardware deps for dry runs.
    from core.system_builder import build_actuator, build_mechanism

    params = PhysicalParams(
        plant=PlantParams(g=9.81, com_length=0.1, tau=0.04, zeta=0.7, num_states=8, max_acc=9.81 * 3),
        workspace=WorkspaceParams(x_ref=0.0, y_ref=0.0, safe_radius=DEFAULT_WORKSPACE.safe_radius),
        mechanism=DEFAULT_MECHANISM,
        hardware=None,
        run=RunParams(),
    )
    params.workspace.safe_radius = DEFAULT_WORKSPACE.safe_radius
    mech = build_mechanism(params)
    actuator = build_actuator(params, mech)
    from hardware.servos.Servo_System import ServoSystem

    return ServoSystem(mech, port=port, frequency=250)


def _drain_events(reader: DVSReader) -> np.ndarray | None:
    batches = []
    while True:
        b = reader.get_event_batch()
        if b is None or len(b) == 0:
            break
        batches.append(b)
    if not batches:
        return None
    return np.concatenate(batches)


def _collect_alignment_for_target(
    *,
    reader: DVSReader,
    mask_y: int,
    title_prefix: str,
    target_str: str,
    decay_display: float,
    surface_intensity_gain: float,
    display_period: float,
    step_x: float,
    step_s: float,
    initial_state: ManualLineState,
) -> tuple[ManualLineState | None, bool]:
    """
    Legacy full-regression: WASD + Space. Returns (state, quit_requested).
    state=None if user quits.
    """
    W, H = DAVIS346_WIDTH, DAVIS346_HEIGHT
    surface = np.zeros((H, W), dtype=np.float32)
    state = ManualLineState(slope_px=initial_state.slope_px, x_at_mask_px=initial_state.x_at_mask_px)

    WINDOW_NAME = "Simple DVS regression calibrator"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    w, h = get_default_window_size(has_cams=True, has_workspace=False)
    cv2.resizeWindow(WINDOW_NAME, w, h)

    next_display = time.perf_counter()
    while reader.is_running():
        ev = _drain_events(reader)
        if ev is not None:
            ev = mask_events_below_line(ev, mask_line_y=mask_y, frame_height=H)
            surface *= decay_display
            if len(ev) > 0:
                np.add.at(surface, (ev["y"], ev["x"]), 1.0)
        else:
            time.sleep(0.0002)

        now = time.perf_counter()
        if now < next_display:
            continue

        frame = np.clip(surface * surface_intensity_gain, 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        draw_manual_overlay(frame, state, mask_y=mask_y)

        x_at_mask = float(state.x_at_mask_px)
        s_px = float(state.slope_px)
        title = f"{title_prefix} | {target_str} | s={s_px:+.4f}, x_at_mask={x_at_mask:+.1f} | WASD adjust, SPACE save, Q quit"
        composite = build_composite(title, frame, frame, None)
        cv2.imshow(WINDOW_NAME, composite)

        if _window_closed(WINDOW_NAME):
            return None, True

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return None, True
        if key == ord(" "):
            return state, False

        if key == ord("a"):
            state.x_at_mask_px -= step_x
        elif key == ord("d"):
            state.x_at_mask_px += step_x
        elif key == ord("w"):
            state.slope_px -= step_s
        elif key == ord("s"):
            state.slope_px += step_s

        state.x_at_mask_px = float(max(0.0, min(W - 1.0, state.x_at_mask_px)))

        while next_display <= now:
            next_display += display_period

    return None, True


def main() -> None:
    args = parse_args()

    if args.full_regression:
        if args.cam1 is not None and args.cam2 is not None:
            device1, device2 = args.cam1, args.cam2
        elif args.cam1 is not None or args.cam2 is not None:
            raise SystemExit("Provide both --cam1 and --cam2, or omit both for discovery.")
        else:
            devices = discover_devices()
            if len(devices) < 2:
                raise SystemExit("Need at least 2 DVS cameras connected.")
            device1, device2 = devices[0], devices[1]

        noise_ms = args.noise_filter_duration
        reader1 = DVSReader(device1, noise_filter_duration_ms=noise_ms)
        reader2 = DVSReader(device2, noise_filter_duration_ms=noise_ms)

        actuator = _build_optional_actuator(args.port)

        r = float(args.workspace_radius)
        x_vals = np.linspace(-0.8 * r, 0.8 * r, int(args.n_position_points))
        y_vals = np.linspace(-0.8 * r, 0.8 * r, int(args.n_position_points))

        tilt_degs = np.arange(args.tilt_deg_min, args.tilt_deg_max + 1e-9, args.tilt_deg_step, dtype=float)
        tilt_rads = np.deg2rad(tilt_degs)

        decay = float(args.decay_display)
        gain = float(args.surface_intensity_gain)
        display_period = 1.0 / float(args.display_fps)

        init_state_cam1 = ManualLineState(slope_px=0.0, x_at_mask_px=DAVIS346_WIDTH / 2)
        init_state_cam2 = ManualLineState(slope_px=0.0, x_at_mask_px=DAVIS346_WIDTH / 2)

        try:
            x_at_mask_samples_1: list[float] = []
            X_true_samples: list[float] = []

            for i, X in enumerate(x_vals):
                if actuator is not None:
                    actuator.send(TableCommand(x_des=float(X), y_des=0.0))
                    time.sleep(args.settle)
                target = f"cam1 position {i+1}/{len(x_vals)} (upright): set X={X:+.4f} m, Y=+0.0000 m, then align"
                st, quit_req = _collect_alignment_for_target(
                    reader=reader1,
                    mask_y=int(args.mask_y_cam1),
                    title_prefix="Cam1",
                    target_str=target,
                    decay_display=decay,
                    surface_intensity_gain=gain,
                    display_period=display_period,
                    step_x=float(args.step_x),
                    step_s=float(args.step_s),
                    initial_state=init_state_cam1,
                )
                if quit_req or st is None:
                    return
                init_state_cam1 = st
                x_at_mask_samples_1.append(float(st.x_at_mask_px))
                X_true_samples.append(float(X))

            kx, bx = fit_affine(np.array(x_at_mask_samples_1), np.array(X_true_samples))

            slope_samples_1: list[float] = []
            ax_true_samples: list[float] = []

            for i, (deg, ax) in enumerate(zip(tilt_degs, tilt_rads)):
                target = f"cam1 tilt {i+1}/{len(tilt_degs)}: set alpha_x={deg:+.1f} deg at (X=0,Y=0), then align"
                st, quit_req = _collect_alignment_for_target(
                    reader=reader1,
                    mask_y=int(args.mask_y_cam1),
                    title_prefix="Cam1",
                    target_str=target,
                    decay_display=decay,
                    surface_intensity_gain=gain,
                    display_period=display_period,
                    step_x=float(args.step_x),
                    step_s=float(args.step_s),
                    initial_state=init_state_cam1,
                )
                if quit_req or st is None:
                    return
                init_state_cam1 = st
                slope_samples_1.append(float(st.slope_px))
                ax_true_samples.append(float(ax))

            kax, bax = fit_affine(np.array(slope_samples_1), np.array(ax_true_samples))

            cam1_cal = SimpleDVSCameraCalibration(k_pos=kx, b_pos=bx, k_alpha=kax, b_alpha=bax)

            x_at_mask_samples_2: list[float] = []
            Y_true_samples: list[float] = []

            for i, Y in enumerate(y_vals):
                if actuator is not None:
                    actuator.send(TableCommand(x_des=0.0, y_des=float(Y)))
                    time.sleep(args.settle)
                target = f"cam2 position {i+1}/{len(y_vals)} (upright): set X=+0.0000 m, Y={Y:+.4f} m, then align"
                st, quit_req = _collect_alignment_for_target(
                    reader=reader2,
                    mask_y=int(args.mask_y_cam2),
                    title_prefix="Cam2",
                    target_str=target,
                    decay_display=decay,
                    surface_intensity_gain=gain,
                    display_period=display_period,
                    step_x=float(args.step_x),
                    step_s=float(args.step_s),
                    initial_state=init_state_cam2,
                )
                if quit_req or st is None:
                    return
                init_state_cam2 = st
                x_at_mask_samples_2.append(float(st.x_at_mask_px))
                Y_true_samples.append(float(Y))

            ky, by = fit_affine(np.array(x_at_mask_samples_2), np.array(Y_true_samples))

            slope_samples_2: list[float] = []
            ay_true_samples: list[float] = []

            for i, (deg, ay) in enumerate(zip(tilt_degs, tilt_rads)):
                target = f"cam2 tilt {i+1}/{len(tilt_degs)}: set alpha_y={deg:+.1f} deg at (X=0,Y=0), then align"
                st, quit_req = _collect_alignment_for_target(
                    reader=reader2,
                    mask_y=int(args.mask_y_cam2),
                    title_prefix="Cam2",
                    target_str=target,
                    decay_display=decay,
                    surface_intensity_gain=gain,
                    display_period=display_period,
                    step_x=float(args.step_x),
                    step_s=float(args.step_s),
                    initial_state=init_state_cam2,
                )
                if quit_req or st is None:
                    return
                init_state_cam2 = st
                slope_samples_2.append(float(st.slope_px))
                ay_true_samples.append(float(ay))

            kay, bay = fit_affine(np.array(slope_samples_2), np.array(ay_true_samples))

            cam2_cal = SimpleDVSCameraCalibration(k_pos=ky, b_pos=by, k_alpha=kay, b_alpha=bay)

            model = SimpleDVSRegressionModel(
                cam1=cam1_cal,
                cam2=cam2_cal,
                mask_y_cam1=int(args.mask_y_cam1),
                mask_y_cam2=int(args.mask_y_cam2),
                metadata={
                    "position_points": int(args.n_position_points),
                    "tilt_deg_min": float(args.tilt_deg_min),
                    "tilt_deg_max": float(args.tilt_deg_max),
                    "tilt_deg_step": float(args.tilt_deg_step),
                    "notes": "cam1: X/alpha_x, cam2: Y/alpha_y; affine fits",
                },
            )
            out_path = Path(args.output)
            model.save(out_path)
            print(f"Saved simple regression model to {out_path}")

        finally:
            reader1.close()
            reader2.close()
            if actuator is not None:
                try:
                    actuator.close()
                except Exception:
                    pass
            cv2.destroyAllWindows()
        return

    # ------------------------------------------------------------------
    # Default: B1 then B2 (two cameras, shared OneDvsVisualizer)
    # ------------------------------------------------------------------
    if args.cam1 is not None and args.cam2 is not None:
        device1, device2 = args.cam1, args.cam2
    elif args.cam1 is None and args.cam2 is None:
        devices = discover_devices()
        if len(devices) < 2:
            raise SystemExit("Need at least 2 DVS cameras for B1+B2 calibration.")
        device1, device2 = devices[0], devices[1]
    else:
        raise SystemExit("Provide both --cam1 and --cam2, or omit both for discovery.")

    noise_ms = args.noise_filter_duration
    reader1 = DVSReader(device1, noise_filter_duration_ms=noise_ms)
    reader2 = DVSReader(device2, noise_filter_duration_ms=noise_ms)
    actuator = _build_optional_actuator(args.port)

    r = float(args.workspace_radius)
    x_step = float(args.x_step_m)
    position_targets = x_positions_from_safe_radius(r, step_m=x_step)

    decay = float(args.decay_display)
    gain = float(args.surface_intensity_gain)
    display_period = 1.0 / float(args.display_fps)

    W, H = DAVIS346_WIDTH, DAVIS346_HEIGHT
    cam_model = CameraModel(width=W, height=H)
    viz = OneDvsVisualizer(
        cam_index=0,
        width=W,
        height=H,
        event_frames_fn=None,
        surface_gain=gain,
    )

    init_cam1 = ManualLineState(slope_px=0.0, x_at_mask_px=DAVIS346_WIDTH / 2)
    init_cam2 = ManualLineState(slope_px=0.0, x_at_mask_px=DAVIS346_WIDTH / 2)

    b1 = B1InterceptCalibration(
        reader=reader1,
        mask_y=int(args.mask_y_cam1),
        viz=viz,
        cam_model=cam_model,
        position_targets_m=position_targets,
        decay_display=decay,
        display_period=display_period,
        step_x=float(args.step_x),
        initial_state=init_cam1,
        device_id=str(device1),
        safe_radius_m=r,
        x_step_m=x_step,
        settle_s=float(args.settle),
        actuator=actuator,
    )

    b2 = B2InterceptCalibration(
        reader=reader2,
        mask_y=int(args.mask_y_cam2),
        viz=viz,
        cam_model=cam_model,
        position_targets_m=position_targets,
        decay_display=decay,
        display_period=display_period,
        step_x=float(args.step_x),
        initial_state=init_cam2,
        device_id=str(device2),
        safe_radius_m=r,
        x_step_m=x_step,
        settle_s=float(args.settle),
        actuator=actuator,
    )

    tilt_degs = list(DEFAULT_TILT_CALIB_DEGS)
    tilt_rads = tilt_degs_to_rads(tilt_degs)
    cx = float(W) / 2.0
    init_s_tilt = ManualLineState(slope_px=0.0, x_at_mask_px=cx)

    s1 = S1SlopeCalibration(
        reader=reader1,
        mask_y=int(args.mask_y_cam1),
        viz=viz,
        cam_model=cam_model,
        angle_targets_rad=tilt_rads,
        decay_display=decay,
        display_period=display_period,
        step_s=float(args.step_s),
        initial_state=init_s_tilt,
        device_id=str(device1),
        settle_s=float(args.settle),
        actuator=actuator,
        center_x_at_mask_px=cx,
    )

    s2 = S2SlopeCalibration(
        reader=reader2,
        mask_y=int(args.mask_y_cam2),
        viz=viz,
        cam_model=cam_model,
        angle_targets_rad=tilt_rads,
        decay_display=decay,
        display_period=display_period,
        step_s=float(args.step_s),
        initial_state=ManualLineState(slope_px=0.0, x_at_mask_px=cx),
        device_id=str(device2),
        settle_s=float(args.settle),
        actuator=actuator,
        center_x_at_mask_px=cx,
    )

    try:
        if not b1.run():
            print("Calibration aborted during B1.")
            return
        if not b2.run():
            print("Calibration aborted during B2.")
            return
        if not s1.run():
            print("Calibration aborted during S1.")
            return
        if not s2.run():
            print("Calibration aborted during S2.")
            return
        out_ds = Path(args.dataset_output)
        save_calibration_dataset_json(
            out_ds,
            b1_samples=b1.samples,
            b2_samples=b2.samples,
            s1_samples=s1.samples,
            s2_samples=s2.samples,
            x_targets_m=position_targets,
            tilt_targets_deg=tilt_degs,
            device_id_cam1=str(device1),
            device_id_cam2=str(device2),
            mask_y_cam1=int(args.mask_y_cam1),
            mask_y_cam2=int(args.mask_y_cam2),
            safe_radius_m=r,
            x_step_m=x_step,
            metadata={
                "notes": (
                    "B: b vs position; S: slope vs tilt at ref. "
                    "S stages: A/D slope, x@mask centered."
                )
            },
        )
        print(
            f"Saved dataset ({len(b1.samples)} b1, {len(b2.samples)} b2, "
            f"{len(s1.samples)} s1, {len(s2.samples)} s2) to {out_ds}"
        )
    finally:
        reader1.close()
        reader2.close()
        if actuator is not None:
            try:
                actuator.close()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
