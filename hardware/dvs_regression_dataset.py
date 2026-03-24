"""
Grid × tilt DVS regression dataset tool.

Builds a grid of points inside the workspace circle, and for each grid point
and each tilt condition (0, +ax, -ax, +ay, -ay) runs:
    step → wait → collect multiple line fits → average → save row

Each dataset row stores:
    [b1, s1, b2, s2, X_true, Y_true, ax_true, ay_true]

The dataset is saved as:
    - perception/calibration_files/dvs_pose_dataset.npz
    - perception/calibration_files/dvs_pose_dataset_meta.json

Usage:
    sudo .venv/bin/python -m hardware.dvs_regression_dataset
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualization.composite_layout import build_composite, get_default_window_size
from core.sim_types import (
    HardwareParams,
    HoughTrackerParams,
    MechanismParams,
    PhysicalParams,
    PlantParams,
    RunParams,
    TableCommand,
    WorkspaceParams,
)
from perception.camera_model import CameraModel
from perception.dvs_camera_reader import (
    DAVIS346_HEIGHT,
    DAVIS346_WIDTH,
    DVSReader,
    discover_devices,
)
from perception.dvs_algorithms import PaperHoughLineAlgorithm, SamLineAlgorithm
from core.system_builder import build_actuator, build_mechanism


DEFAULT_WORKSPACE = WorkspaceParams(x_ref=0.0, y_ref=0.0, safe_radius=0.108)
DEFAULT_MECHANISM = MechanismParams(
    O=(128.77, 178.13),
    B=(101.77, 210.13),
    la=175,
    lb=175,
)


TILT_CONDITIONS = [
    ("0", 0.0, 0.0),
    ("+ax", +1.0, 0.0),
    ("-ax", -1.0, 0.0),
    ("+ay", 0.0, +1.0),
    ("-ay", 0.0, -1.0),
]


def build_grid_points(
    workspace: WorkspaceParams,
    grid_side: int | None = None,
    grid_n: int | None = None,
) -> list[tuple[float, float]]:
    if workspace.safe_radius is None:
        raise ValueError("Workspace safe_radius must be set for calibration.")

    r = workspace.safe_radius

    # finds num of points in the grid, either by side length or by total number of points
    if grid_side is not None:
        nx = ny = grid_side
    elif grid_n is not None:
        nx = ny = max(1, int(round(grid_n**0.5)))
    else:
        nx = ny = 5

    # Creates a grid of points like a square with an inscribed circle 
    xs = np.linspace(-r, r, nx)
    ys = np.linspace(-r, r, ny)

    # since the circle is incribed, not all points will be inside the circle.
    points: list[tuple[float, float]] = []
    for xi in xs:
        for yi in ys:
            if xi * xi + yi * yi <= r * r:
                points.append((float(xi), float(yi)))
    return points


def render_workspace_calibration(
    workspace: WorkspaceParams,
    grid_points: list[tuple[float, float]],
    fully_saved_indices: set[int],
    current_index: int | None,
    next_index: int | None = None,
    workspace_size: int = 350,
    grid_step_m: float = 0.02,
    message: str | None = None,
) -> np.ndarray:
    center = workspace_size // 2
    margin = 20
    if workspace.safe_radius is not None:
        scale = (workspace_size - 2 * margin) / (2 * workspace.safe_radius)
    else:
        scale = 4000.0
    x_ref = workspace.x_ref
    y_ref = workspace.y_ref
    safe_radius = workspace.safe_radius

    canvas = np.zeros((workspace_size, workspace_size), dtype=np.uint8)
    canvas[:] = 40
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    grid_color = (55, 55, 55)
    circle_color = (100, 100, 100)

    if safe_radius is not None:
        n_grid = int(np.ceil(safe_radius / grid_step_m))
        for k in range(-n_grid, n_grid + 1):
            x_world = x_ref + k * grid_step_m
            px = int(center + (x_world - x_ref) * scale)
            if 0 <= px < workspace_size:
                cv2.line(canvas, (px, 0), (px, workspace_size - 1), grid_color, 1)
            y_world = y_ref + k * grid_step_m
            py = int(center - (y_world - y_ref) * scale)
            if 0 <= py < workspace_size:
                cv2.line(canvas, (0, py), (workspace_size - 1, py), grid_color, 1)
        radius_px = int(safe_radius * scale)
        cv2.circle(canvas, (center, center), radius_px, circle_color, 1)

    cross_len = 15
    cv2.line(canvas, (center - cross_len, center), (center + cross_len, center), circle_color, 1)
    cv2.line(canvas, (center, center - cross_len), (center, center + cross_len), circle_color, 1)

    def world_to_px(dx: float, dy: float) -> tuple[int, int]:
        x_w = x_ref + dx
        y_w = y_ref + dy
        px = int(center + (x_w - x_ref) * scale)
        py = int(center - (y_w - y_ref) * scale)
        return (px, py)

    for i, (dx, dy) in enumerate(grid_points):
        px, py = world_to_px(dx, dy)
        if 0 <= px < workspace_size and 0 <= py < workspace_size:
            if i == current_index:
                color = (255, 0, 0)
            elif i in fully_saved_indices:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.circle(canvas, (px, py), 4, color, -1)

    if (
        current_index is not None
        and next_index is not None
        and 0 <= next_index < len(grid_points)
        and current_index != next_index
    ):
        p1 = world_to_px(grid_points[current_index][0], grid_points[current_index][1])
        p2 = world_to_px(grid_points[next_index][0], grid_points[next_index][1])
        if all(0 <= p[0] < workspace_size and 0 <= p[1] < workspace_size for p in (p1, p2)):
            cv2.arrowedLine(canvas, p1, p2, (255, 255, 0), 2, tipLength=0.2)

    if message:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(message, font, 0.6, 2)
        tx = (workspace_size - tw) // 2
        ty = workspace_size - 25
        cv2.putText(canvas, message, (tx, ty), font, 0.6, (255, 255, 255), 2)

    return canvas


def render_round_intro(
    workspace_size: int = 350,
    round_index: int = 0,
    total_rounds: int = 5,
    tilt_label: str = "0",
    tilt_angle_deg: float = 0.0,
) -> np.ndarray:
    """Darkened workspace panel with round number and tilt instruction. Wait for SPACE to begin."""
    canvas = np.zeros((workspace_size, workspace_size), dtype=np.uint8)
    canvas[:] = 15
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 28
    cy = workspace_size // 2 - (2 * line_height)

    round_text = f"Round {round_index + 1} of {total_rounds}"
    (tw, th), _ = cv2.getTextSize(round_text, font, 0.7, 2)
    tx = (workspace_size - tw) // 2
    cv2.putText(canvas, round_text, (tx, cy), font, 0.7, (200, 200, 200), 2)
    cy += line_height

    if tilt_label == "0":
        tilt_text = "Set pencil tilt: 0 (upright)"
    else:
        tilt_text = f"Set pencil tilt: {tilt_label} ({tilt_angle_deg:.0f} deg)"
    (tw, th), _ = cv2.getTextSize(tilt_text, font, 0.6, 2)
    tx = (workspace_size - tw) // 2
    cv2.putText(canvas, tilt_text, (tx, cy), font, 0.6, (220, 220, 220), 2)
    cy += line_height

    prompt = "Press SPACE to begin"
    (tw, th), _ = cv2.getTextSize(prompt, font, 0.6, 2)
    tx = (workspace_size - tw) // 2
    cv2.putText(canvas, prompt, (tx, cy), font, 0.6, (255, 255, 255), 2)

    return canvas


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DVS regression dataset collection: grid × tilt → averaged (b,s) measurements."
    )
    p.add_argument("--cam1", help="Camera 1 serial or device (omit for discovery)")
    p.add_argument("--cam2", help="Camera 2 serial or device (omit for discovery)")
    p.add_argument("--grid", type=int, default=3, metavar="N", help="Grid side (N×N points inside circle)")
    p.add_argument(
        "--grid-n", type=int, default=None, metavar="N", help="Target total points (overrides --grid side)"
    )
    p.add_argument("--settle", type=float, default=2.0, metavar="SEC", help="Settle time after each step (seconds)")
    p.add_argument(
        "--frames-per-pose",
        type=int,
        default=300,
        metavar="N",
        help="Target number of valid line fits to average per pose",
    )
    p.add_argument(
        "--max-collect-time",
        type=float,
        default=5.0,
        metavar="SEC",
        help="Max collection time per pose (seconds)",
    )
    p.add_argument(
        "--tilt-angle-deg",
        type=float,
        default=10.0,
        metavar="DEG",
        help="Nominal tilt magnitude (deg) for ±ax/±ay conditions",
    )
    p.add_argument(
        "--output-npz",
        type=str,
        default="perception/calibration_files/dvs_pose_dataset.npz",
        help="Output .npz path for dataset array",
    )
    p.add_argument(
        "--output-meta",
        type=str,
        default="perception/calibration_files/dvs_pose_dataset_meta.json",
        help="Output JSON path for dataset metadata",
    )
    p.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Servo serial port (or None for mock)")
    p.add_argument(
        "--workspace-radius",
        type=float,
        default=DEFAULT_WORKSPACE.safe_radius,
        help="Workspace safe_radius (m)",
    )
    p.add_argument("--mode", choices=["hough", "sam"], default="hough", help="Line algorithm")
    p.add_argument(
        "--noise-filter-duration",
        type=float,
        default=30.0,
        metavar="MS",
        help="Event noise filter ms (Sam)",
    )
    p.add_argument("--display-fps", type=float, default=30.0, help="Display refresh rate")
    p.add_argument("--decay-display", type=float, default=0.5, help="Event surface decay")
    p.add_argument("--surface-intensity-gain", type=float, default=50.0, help="Surface brightness")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.cam1 is not None and args.cam2 is not None:
        device1, device2 = args.cam1, args.cam2
        print("Opening cameras (from serials)...")
    elif args.cam1 is not None or args.cam2 is not None:
        print("Error: Provide both --cam1 and --cam2, or omit both.", file=sys.stderr)
        sys.exit(1)
    else:
        devices = discover_devices()
        print(f"Discovered devices: {devices}")
        if len(devices) < 2:
            print("Error: Need at least 2 DVS cameras.", file=sys.stderr)
            sys.exit(1)
        device1, device2 = devices[0], devices[1]
        print("Using devices[0] and devices[1] for x and y cams.")

    workspace = WorkspaceParams(
        x_ref=DEFAULT_WORKSPACE.x_ref,
        y_ref=DEFAULT_WORKSPACE.y_ref,
        safe_radius=args.workspace_radius,
    )
    grid_points = build_grid_points(
        workspace,
        grid_side=args.grid if args.grid_n is None else None,
        grid_n=args.grid_n,
    )
    if not grid_points:
        print("Error: No grid points inside workspace circle.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Regression dataset mode - grid × tilt. "
        f"Points inside workspace: {len(grid_points)}, tilts: {len(TILT_CONDITIONS)}. Press Q to abort."
    )

    params = PhysicalParams(
        plant=PlantParams(
            g=9.81,
            com_length=0.1,
            tau=0.04,
            zeta=0.7,
            num_states=8,
            max_acc=9.81 * 3,
        ),
        workspace=workspace,
        mechanism=DEFAULT_MECHANISM,
        hardware=HardwareParams(
            servo=True,
            servo_port=None if args.port.strip().lower() == "none" else args.port,
            servo_frequency=250,
            dvs_cam=False,
        ),
        run=RunParams(),
    )
    mech = build_mechanism(params)
    actuator = build_actuator(params, mech)
    if actuator is None:
        print("Error: Actuator not built (servo disabled?). Use --port for real hardware.", file=sys.stderr)
        sys.exit(1)

    noise_ms = args.noise_filter_duration if args.mode == "sam" else None
    reader1 = DVSReader(device1, noise_filter_duration_ms=noise_ms)
    reader2 = DVSReader(device2, noise_filter_duration_ms=noise_ms)

    W, H = DAVIS346_WIDTH, DAVIS346_HEIGHT
    if args.mode == "hough":
        hp = HoughTrackerParams()
        algo1 = PaperHoughLineAlgorithm(width=W, height=H, params=hp)
        algo2 = PaperHoughLineAlgorithm(width=W, height=H, params=hp)
    else:
        algo1 = SamLineAlgorithm(width=W, height=H, min_points=50)
        algo2 = SamLineAlgorithm(width=W, height=H, min_points=50)
    cam_model = CameraModel(width=W, height=H)

    WINDOW_NAME = "DVS Regression Dataset"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    w, h = get_default_window_size(has_cams=True, has_workspace=True)
    cv2.resizeWindow(WINDOW_NAME, w, h)

    decay = args.decay_display
    gain = args.surface_intensity_gain
    display_period = 1.0 / args.display_fps
    surface1 = np.zeros((H, W), dtype=np.float32)
    surface2 = np.zeros((H, W), dtype=np.float32)
    result1, result2 = None, None

    data_rows: list[list[float]] = []
    saved_this_round: set[int] = set()
    next_display = time.perf_counter()
    tilt_angle_rad = math.radians(args.tilt_angle_deg)

    def drain_and_draw_cameras():
        nonlocal surface1, surface2, result1, result2, next_display
        batches1, batches2 = [], []
        while True:
            b = reader1.get_event_batch()
            if b is None or len(b) == 0:
                break
            batches1.append(b)
        while True:
            b = reader2.get_event_batch()
            if b is None or len(b) == 0:
                break
            batches2.append(b)
        if batches1:
            ev1 = np.concatenate(batches1)
            surface1 *= decay
            np.add.at(surface1, (ev1["y"], ev1["x"]), 1.0)
            result1 = algo1.update(ev1)
        if batches2:
            ev2 = np.concatenate(batches2)
            surface2 *= decay
            np.add.at(surface2, (ev2["y"], ev2["x"]), 1.0)
            result2 = algo2.update(ev2)
        frame1 = np.clip(surface1 * gain, 0, 255).astype(np.uint8)
        frame2 = np.clip(surface2 * gain, 0, 255).astype(np.uint8)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)
        for frame, res in [(frame1, result1), (frame2, result2)]:
            if res is not None and not isinstance(res, tuple):
                s_px, b_px = res.slope, res.intercept
                y0, y1 = 0, H - 1
                x0 = int(s_px * y0 + b_px)
                x1 = int(s_px * y1 + b_px)
                cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return frame1, frame2

    print("Press SPACE to begin dataset collection.")
    started = False
    while not started:
        frame1, frame2 = drain_and_draw_cameras()
        ws_canvas = render_workspace_calibration(
            workspace,
            grid_points,
            set(),
            None,
            next_index=0,
            message="Press SPACE to begin",
        )
        title = "Regression dataset - Press SPACE to begin | Q: abort"
        composite = build_composite(title, frame1, frame2, ws_canvas)
        cv2.imshow(WINDOW_NAME, composite)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            started = True
        elif key == ord("q"):
            print("Aborted by user (Q).")
            reader1.close()
            reader2.close()
            cv2.destroyAllWindows()
            return

    try:
        for tilt_idx, (tilt_label, ax_sign, ay_sign) in enumerate(TILT_CONDITIONS):
            saved_this_round.clear()

            # Round intro: darkened workspace, show round and tilt, wait for SPACE
            print(f"Round {tilt_idx + 1}/{len(TILT_CONDITIONS)}: Set tilt to {tilt_label}, then press SPACE.")
            round_started = False
            while not round_started:
                frame1, frame2 = drain_and_draw_cameras()
                ws_canvas = render_round_intro(
                    workspace_size=350,
                    round_index=tilt_idx,
                    total_rounds=len(TILT_CONDITIONS),
                    tilt_label=tilt_label,
                    tilt_angle_deg=args.tilt_angle_deg,
                )
                title = f"Round {tilt_idx + 1}/{len(TILT_CONDITIONS)} - Tilt {tilt_label} | SPACE: begin, Q: abort"
                composite = build_composite(title, frame1, frame2, ws_canvas)
                cv2.imshow(WINDOW_NAME, composite)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    round_started = True
                elif key == ord("q"):
                    print("Aborted by user (Q).")
                    return

            # Run all grid positions for this tilt: move → settle → collect
            for idx, (dx, dy) in enumerate(grid_points):
                x_cmd = workspace.x_ref + dx
                y_cmd = workspace.y_ref + dy
                cmd = TableCommand(x_des=x_cmd, y_des=y_cmd)
                actuator.send(cmd)

                t_end_settle = time.perf_counter() + args.settle
                while time.perf_counter() < t_end_settle:
                    frame1, frame2 = drain_and_draw_cameras()
                    now = time.perf_counter()
                    if now >= next_display:
                        next_idx = idx + 1 if idx + 1 < len(grid_points) else None
                        msg = f"Tilt {tilt_label} - settling ({args.settle - (t_end_settle - now):.1f}s)"
                        ws_canvas = render_workspace_calibration(
                            workspace,
                            grid_points,
                            saved_this_round,
                            idx,
                            next_index=next_idx,
                            message=msg,
                        )
                        title = (
                            f"Round {tilt_idx + 1}/{len(TILT_CONDITIONS)} - Tilt {tilt_label} | "
                            f"Point {idx + 1}/{len(grid_points)} | Q: abort"
                        )
                        composite = build_composite(title, frame1, frame2, ws_canvas)
                        cv2.imshow(WINDOW_NAME, composite)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("Aborted by user (Q).")
                            return
                        while next_display <= now:
                            next_display += display_period
                    else:
                        time.sleep(0.001)

                measurements_b1: list[float] = []
                measurements_s1: list[float] = []
                measurements_b2: list[float] = []
                measurements_s2: list[float] = []
                collect_start = time.perf_counter()

                while True:
                    frame1, frame2 = drain_and_draw_cameras()

                    if (
                        result1 is not None
                        and not isinstance(result1, tuple)
                        and result2 is not None
                        and not isinstance(result2, tuple)
                    ):
                        obs1 = cam_model.pixel_to_camnorm(result1)
                        obs2 = cam_model.pixel_to_camnorm(result2)
                        measurements_b1.append(float(obs1.intercept))
                        measurements_s1.append(float(obs1.slope))
                        measurements_b2.append(float(obs2.intercept))
                        measurements_s2.append(float(obs2.slope))

                    now = time.perf_counter()
                    if now >= next_display:
                        next_idx = idx + 1 if idx + 1 < len(grid_points) else None
                        msg = (
                            f"Tilt {tilt_label} - collecting "
                            f"({len(measurements_b1)}/{args.frames_per_pose} frames)"
                        )
                        ws_canvas = render_workspace_calibration(
                            workspace,
                            grid_points,
                            saved_this_round,
                            idx,
                            next_index=next_idx,
                            message=msg,
                        )
                        title = (
                            f"Round {tilt_idx + 1}/{len(TILT_CONDITIONS)} - Tilt {tilt_label} | "
                            f"Point {idx + 1}/{len(grid_points)} | Q: abort"
                        )
                        composite = build_composite(title, frame1, frame2, ws_canvas)
                        cv2.imshow(WINDOW_NAME, composite)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("Aborted by user (Q).")
                            return
                        while next_display <= now:
                            next_display += display_period
                    else:
                        time.sleep(0.001)

                    if len(measurements_b1) >= args.frames_per_pose:
                        break
                    if now - collect_start > args.max_collect_time:
                        break

                if not measurements_b1:
                    print(
                        f"Point {idx + 1}/{len(grid_points)}, tilt {tilt_label}: "
                        "no valid line fits, skipping."
                    )
                    continue

                b1 = float(np.mean(measurements_b1))
                s1 = float(np.mean(measurements_s1))
                b2 = float(np.mean(measurements_b2))
                s2 = float(np.mean(measurements_s2))
                ax_true = ax_sign * tilt_angle_rad
                ay_true = ay_sign * tilt_angle_rad

                data_rows.append(
                    [b1, s1, b2, s2, float(x_cmd), float(y_cmd), float(ax_true), float(ay_true)]
                )
                saved_this_round.add(idx)

                print(
                    f"Recorded: round {tilt_idx + 1} tilt {tilt_label}, point {idx + 1}/{len(grid_points)}, "
                    f"x_cmd={x_cmd:.4f}, y_cmd={y_cmd:.4f}, frames={len(measurements_b1)}"
                )

        if not data_rows:
            print("No samples recorded; nothing to save.")
            return

        data = np.asarray(data_rows, dtype=np.float64)
        out_npz_path = Path(args.output_npz)
        out_npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_npz_path, data=data)

        nx = ny = args.grid if args.grid_n is None else int(round(args.grid_n**0.5))
        tilt_angle_deg = float(args.tilt_angle_deg)
        meta = {
            "column_names": [
                "b1",
                "s1",
                "b2",
                "s2",
                "X_true",
                "Y_true",
                "ax_true",
                "ay_true",
            ],
            "grid_shape": [nx, ny],
            "tilt_angles_deg": {
                "0": {"ax": 0.0, "ay": 0.0},
                "+ax": {"ax": +tilt_angle_deg, "ay": 0.0},
                "-ax": {"ax": -tilt_angle_deg, "ay": 0.0},
                "+ay": {"ax": 0.0, "ay": +tilt_angle_deg},
                "-ay": {"ax": 0.0, "ay": -tilt_angle_deg},
            },
            "total_samples": int(data.shape[0]),
            "frames_per_pose_target": int(args.frames_per_pose),
        }
        out_meta_path = Path(args.output_meta)
        out_meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(
            f"Saved regression dataset with {data.shape[0]} samples to {out_npz_path} "
            f"and metadata to {out_meta_path}."
        )

    finally:
        reader1.close()
        reader2.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

