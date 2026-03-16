"""
Grid-sweep DVS calibration tool.

Builds a grid of points inside the workspace circle, runs step→wait→record for each
point, and saves (x_cmd, y_cmd, b1, b2) to JSON for runtime 2D interpolation.

Usage:
    python -m hardware.dvs_camera_calibration
    python -m hardware.dvs_camera_calibration --grid 7 --settle 2.0 --cam1 SERIAL1 --cam2 SERIAL2

Requires servo calibration first; then run this with DVS cams and mechanism connected.
"""

import argparse
import json
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
from system_builder import build_actuator, build_mechanism


DEFAULT_WORKSPACE = WorkspaceParams(x_ref=0.0, y_ref=0.0, safe_radius=0.108)
DEFAULT_MECHANISM = MechanismParams(
    O=(128.77, 178.13),
    B=(101.77, 210.13),
    la=175,
    lb=175,
)


def build_grid_points(
    workspace: WorkspaceParams,
    grid_side: int | None = None,
    grid_n: int | None = None,
) -> list[tuple[float, float]]:
    """Points (dx, dy) in ref-centered coords inside the workspace circle."""
    if workspace.safe_radius is None:
        raise ValueError("Workspace safe_radius must be set for calibration.")
    r = workspace.safe_radius
    if grid_side is not None:
        nx = ny = grid_side
    elif grid_n is not None:
        nx = ny = max(1, int(round(grid_n ** 0.5)))
    else:
        nx = ny = 5
    xs = np.linspace(-r, r, nx)
    ys = np.linspace(-r, r, ny)
    points: list[tuple[float, float]] = []
    for xi in xs:
        for yi in ys:
            if xi * xi + yi * yi <= r * r:
                points.append((float(xi), float(yi)))
    return points


def render_workspace_calibration(
    workspace: WorkspaceParams,
    grid_points: list[tuple[float, float]],
    saved_indices: set[int],
    current_index: int | None,
    next_index: int | None = None,
    workspace_size: int = 350,
    grid_step_m: float = 0.02,
    message: str | None = None,
) -> np.ndarray:
    """Workspace canvas with grid, circle, dots (red pending, green saved, blue current), and optional arrow to next."""
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
            elif i in saved_indices:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.circle(canvas, (px, py), 4, color, -1)

    # Arrow from current (blue) point to next point to be calibrated
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid-sweep DVS calibration: step–wait–record, save JSON.")
    p.add_argument("--cam1", help="Camera 1 serial or device (omit for discovery)")
    p.add_argument("--cam2", help="Camera 2 serial or device (omit for discovery)")
    p.add_argument("--grid", type=int, default=3, metavar="N", help="Grid side (N×N points inside circle)")
    p.add_argument("--grid-n", type=int, default=None, metavar="N", help="Target total points (overrides --grid)")
    p.add_argument("--settle", type=float, default=2.0, metavar="SEC", help="Settle time after each step (seconds)")
    p.add_argument("--output", type=str, default="perception/calibration_files/dvs_calibration.json", help="Output JSON path")
    p.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Servo serial port (or None for mock)")
    p.add_argument("--workspace-radius", type=float, default=DEFAULT_WORKSPACE.safe_radius, help="Workspace safe_radius (m)")
    p.add_argument("--mode", choices=["hough", "sam"], default="hough", help="Line algorithm")
    p.add_argument("--noise-filter-duration", type=float, default=30.0, metavar="MS", help="Event noise filter ms (Sam)")
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

    print(f"Calibration mode - grid sweep. Points inside workspace: {len(grid_points)}. Press Q to abort.")

    params = PhysicalParams(
        plant=PlantParams(
            g=9.81, com_length=0.1, tau=0.04, zeta=0.7, num_states=8, max_acc=9.81 * 3,
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

    WINDOW_NAME = "DVS Calibration"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    w, h = get_default_window_size(has_cams=True, has_workspace=True)
    cv2.resizeWindow(WINDOW_NAME, w, h)

    decay = args.decay_display
    gain = args.surface_intensity_gain
    display_period = 1.0 / args.display_fps
    surface1 = np.zeros((H, W), dtype=np.float32)
    surface2 = np.zeros((H, W), dtype=np.float32)
    result1, result2 = None, None

    samples: list[dict] = []
    saved_indices: set[int] = set()
    next_display = time.perf_counter()

    print("Press SPACE to begin calibration.")
    started = False
    while not started:
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
        ws_canvas = render_workspace_calibration(
            workspace, grid_points, saved_indices, None, next_index=0,
            message="Press SPACE to begin",
        )
        title = "Calibration - Press SPACE to begin | Q: abort"
        composite = build_composite(title, frame1, frame2, ws_canvas)
        cv2.imshow(WINDOW_NAME, composite)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            started = True
        elif key == ord("q"):
            print("Aborted by user (Q).")
            return

    try:
        for idx, (dx, dy) in enumerate(grid_points):
            x_cmd = workspace.x_ref + dx
            y_cmd = workspace.y_ref + dy
            cmd = TableCommand(x_des=x_cmd, y_des=y_cmd)
            actuator.send(cmd)

            t_end = time.perf_counter() + args.settle
            while time.perf_counter() < t_end:
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

                now = time.perf_counter()
                if now >= next_display:
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
                    next_idx = idx + 1 if idx + 1 < len(grid_points) else None
                    ws_canvas = render_workspace_calibration(
                        workspace, grid_points, saved_indices, idx, next_index=next_idx,
                    )
                    title = f"Calibration - Point {idx + 1}/{len(grid_points)} | Q: abort"
                    composite = build_composite(title, frame1, frame2, ws_canvas)
                    cv2.imshow(WINDOW_NAME, composite)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Aborted by user (Q).")
                        return
                    while next_display <= now:
                        next_display += display_period
                else:
                    time.sleep(0.001)

            if result1 is not None and not isinstance(result1, tuple) and result2 is not None and not isinstance(result2, tuple):
                obs1 = cam_model.pixel_to_normalized(result1)
                obs2 = cam_model.pixel_to_normalized(result2)
                b1, b2 = obs1.intercept, obs2.intercept
                samples.append({"x": x_cmd, "y": y_cmd, "b1": float(b1), "b2": float(b2)})
                saved_indices.add(idx)
                print(f"Point {idx + 1}/{len(grid_points)} recorded: (x_cmd={x_cmd:.4f}, y_cmd={y_cmd:.4f}) b1={b1:.6f} b2={b2:.6f}")
            else:
                print(f"Point {idx + 1}/{len(grid_points)} skipped (no valid line fit).")

        out_data = {
            "points": samples,
            "x_ref": workspace.x_ref,
            "y_ref": workspace.y_ref,
            "safe_radius": workspace.safe_radius,
            "grid_count": len(grid_points),
            "saved_count": len(samples),
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"Saved {len(samples)} samples to {out_path}.")

    finally:
        reader1.close()
        reader2.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
