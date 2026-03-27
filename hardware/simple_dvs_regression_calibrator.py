"""
Interactive simple DVS regression calibrator.

Workflow (per camera, in order cam1 then cam2):
  1) Position phase (upright): collect 10 samples mapping x_at_mask -> axis position (X for cam1, Y for cam2)
  2) Tilt phase (at reference): collect slope samples mapping slope_px -> alpha (alpha_x for cam1, alpha_y for cam2)

User aligns a manually-controlled overlay line to the pencil using WASD:
  - A/D: move x_at_mask (pixels)
  - W/S: change slope (pixels/pixel)
  - Space: record sample
  - q / Esc: quit

Saves JSON model to `perception/calibration_files/simple_dvs_regression.json` by default.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.sim_types import MechanismParams, PhysicalParams, PlantParams, RunParams, TableCommand, WorkspaceParams
from perception.dvs_algorithms import line_x_at_pixel_y, mask_events_below_line
from perception.dvs_camera_reader import DAVIS346_HEIGHT, DAVIS346_WIDTH, DVSReader, discover_devices
from perception.simple_dvs_regression_model import SimpleDVSCameraCalibration, SimpleDVSRegressionModel, fit_affine
from visualization.composite_layout import build_composite, get_default_window_size


def _window_closed(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


DEFAULT_WORKSPACE = WorkspaceParams(x_ref=0.0, y_ref=0.0, safe_radius=0.108)
DEFAULT_MECHANISM = MechanismParams(
    O=(128.77, 178.13),
    B=(101.77, 210.13),
    la=175,
    lb=175,
)


@dataclass
class ManualLineState:
    slope_px: float = 0.0
    x_at_mask_px: float = 0.0

    def to_obs_px(self, mask_y: int) -> tuple[float, float]:
        b_px = float(self.x_at_mask_px) - float(self.slope_px) * float(mask_y)
        return float(self.slope_px), float(b_px)


def draw_manual_overlay(frame: np.ndarray, state: ManualLineState, mask_y: int) -> None:
    H, W = frame.shape[:2]
    if 0 < mask_y < H:
        cv2.line(frame, (0, mask_y), (W - 1, mask_y), (0, 165, 255), 2)
    s_px, b_px = state.to_obs_px(mask_y=mask_y)
    y0 = 0
    y1 = (min(mask_y - 1, H - 1) if 0 < mask_y < H else (H - 1))
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive simple DVS regression calibrator")
    p.add_argument("--cam1", help="Camera 1 serial/device (omit for discovery)")
    p.add_argument("--cam2", help="Camera 2 serial/device (omit for discovery)")
    p.add_argument("--noise-filter-duration", type=float, default=30.0, metavar="MS", help="Noise filter ms")
    p.add_argument("--mask-y-cam1", type=int, default=160, metavar="Y", help="Mask line y for cam1")
    p.add_argument("--mask-y-cam2", type=int, default=190, metavar="Y", help="Mask line y for cam2")
    p.add_argument("--decay-display", type=float, default=0.5, help="Event surface decay")
    p.add_argument("--surface-intensity-gain", type=float, default=50.0, help="Surface brightness")
    p.add_argument("--display-fps", type=float, default=30.0, help="GUI refresh rate")
    p.add_argument("--workspace-radius", type=float, default=DEFAULT_WORKSPACE.safe_radius, help="Workspace radius (m)")
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
        help="Output JSON model path",
    )
    p.add_argument("--step-x", type=float, default=1.0, help="WASD step for x_at_mask (pixels)")
    p.add_argument("--step-s", type=float, default=0.005, help="WASD step for slope (pixels/pixel)")
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
    # The repo’s build_actuator uses params.hardware; easiest is to bypass and just create ServoSystem,
    # but that pulls in more dependencies. Use dataset tool style instead.
    from hardware.Servo_System import ServoSystem

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
    Returns (state, quit_requested).
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

        # Use WASD for robust cross-platform controls.
        if key == ord("a"):
            state.x_at_mask_px -= step_x
        elif key == ord("d"):
            state.x_at_mask_px += step_x
        elif key == ord("w"):
            state.slope_px -= step_s
        elif key == ord("s"):
            state.slope_px += step_s

        # Clamp x to image bounds for sanity.
        state.x_at_mask_px = float(max(0.0, min(W - 1.0, state.x_at_mask_px)))

        while next_display <= now:
            next_display += display_period

    return None, True


def main() -> None:
    args = parse_args()

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

    # Initial overlay guess: vertical line at center.
    init_state_cam1 = ManualLineState(slope_px=0.0, x_at_mask_px=DAVIS346_WIDTH / 2)
    init_state_cam2 = ManualLineState(slope_px=0.0, x_at_mask_px=DAVIS346_WIDTH / 2)

    try:
        # ------------------------------------------------------------
        # Cam1: X (position) calibration at upright
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # Cam1: alpha_x calibration via tilt sweep at reference
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # Cam2: Y (position) calibration at upright
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # Cam2: alpha_y calibration via tilt sweep at reference
        # ------------------------------------------------------------
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

        # Save model
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


if __name__ == "__main__":
    main()

