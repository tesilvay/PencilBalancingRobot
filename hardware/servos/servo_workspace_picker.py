"""
Standalone tool: connect to servos, show workspace (same view as main with cams+servos),
let the user pick a point by clicking. The point is checked against workspace (x_ref, y_ref, safe_radius),
a TableCommand is generated and sent to the servos. Use this to verify the table moves to the chosen point.

With a real serial port, the same pre-run calibration as main runs first (terminal UI: arrow keys
to align origin, Enter to accept), then the OpenCV workspace picker opens.

Run: python -m hardware.servos.servo_workspace_picker [--port /dev/ttyUSB1]
     Use --port None for mock (no real hardware, no calibration).
"""
import argparse
import numpy as np
import cv2
from core.sim_types import (
    PlantParams,
    WorkspaceParams,
    MechanismParams,
    HardwareParams,
    RunParams,
    PhysicalParams,
    TableCommand,
)
from core.system_builder import build_mechanism, build_actuator
from hardware.servos.servo_workspace_offset_calibrator import calibrate_servo_workspace_offset
from visualization.composite_layout import (
    BANNER_HEIGHT,
    ONE_PANEL_MARGIN,
    build_composite,
    get_default_window_size,
)


def _window_closed(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


# Same defaults as main.py for workspace and mechanism
DEFAULT_WORKSPACE = WorkspaceParams(
    x_ref=0.010,
    y_ref=-0.020,
    safe_radius=0.068,
)
DEFAULT_MECHANISM = MechanismParams(
    O=(128.77, 178.13),
    B=(101.77, 210.13),
    la=175,
    lb=175,
)


def is_inside_workspace(x: float, y: float, workspace: WorkspaceParams) -> bool:
    """True if (x, y) is inside the workspace circle."""
    if workspace.safe_radius is None:
        return True
    dx = x - workspace.x_ref
    dy = y - workspace.y_ref
    return (dx * dx + dy * dy) <= (workspace.safe_radius ** 2)


def clamp_to_workspace(x: float, y: float, workspace: WorkspaceParams) -> tuple[float, float]:
    """Project (x, y) onto workspace circle if outside (same as plant/visualizer)."""
    if workspace.safe_radius is None:
        return x, y
    x_ref = workspace.x_ref
    y_ref = workspace.y_ref
    safe_radius = workspace.safe_radius
    dx = x - x_ref
    dy = y - y_ref
    dist = np.sqrt(dx * dx + dy * dy)
    if dist > safe_radius and dist > 0:
        scale = safe_radius / dist
        dx *= scale
        dy *= scale
        x = x_ref + dx
        y = y_ref + dy
    return x, y


def _render_workspace_canvas(
    workspace: WorkspaceParams,
    command: TableCommand | None,
    workspace_size: int = 350,
    grid_step_m: float = 0.02,
) -> np.ndarray:
    """Draw workspace view (same as main's DVSWorkspaceVisualizer: grid, circle, center cross, target point)."""
    center = workspace_size // 2
    margin = 20
    if workspace.safe_radius is not None:
        scale = (workspace_size - 2 * margin) / (2 * workspace.safe_radius)
    else:
        scale = 4000.0

    canvas = np.zeros((workspace_size, workspace_size), dtype=np.uint8)
    canvas[:] = 40
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    x_ref = workspace.x_ref
    y_ref = workspace.y_ref
    safe_radius = workspace.safe_radius
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

    if command is not None:
        x_des, y_des = clamp_to_workspace(command.x_des, command.y_des, workspace)
        px = int(center + (x_des - x_ref) * scale)
        py = int(center - (y_des - y_ref) * scale)
        if 0 <= px < workspace_size and 0 <= py < workspace_size:
            cv2.circle(canvas, (px, py), 5, (0, 255, 0), -1)

    return canvas, center, scale


def run(
    workspace: WorkspaceParams,
    mechanism_params: MechanismParams,
    servo_port: str | None,
    *,
    skip_calibration: bool = False,
):
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
        mechanism=mechanism_params,
        hardware=HardwareParams(
            servo=True,
            servo_port=servo_port,
            servo_frequency=250,
            dvs_cam=False,
            dvs_cam_x_port=None,
            dvs_cam_y_port=None,
        ),
        run=RunParams(),
    )
    mech = build_mechanism(params)
    actuator = build_actuator(params, mech)
    if actuator is None:
        raise RuntimeError("Actuator not built (servo disabled?)")

    if servo_port is not None and not skip_calibration:
        if not hasattr(actuator, "set_workspace_offset"):
            raise RuntimeError("Actuator must support set_workspace_offset for real-servo calibration.")
        x_offset, y_offset = calibrate_servo_workspace_offset(
            system=None,
            actuator=actuator,
            workspace=workspace,
        )
        actuator.set_workspace_offset(x_offset, y_offset)

    # Same workspace view as main (grid, circle, point) — single composite window with banner
    WINDOW_NAME = "Workspace picker"
    workspace_size = 350
    grid_step_m = 0.02
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    w, h = get_default_window_size(has_cams=False, has_workspace=True)
    cv2.resizeWindow(WINDOW_NAME, w, h)

    current_command: TableCommand | None = None
    center = workspace_size // 2
    scale = (
        (workspace_size - 40) / (2 * workspace.safe_radius)
        if workspace.safe_radius is not None
        else 4000.0
    )

    # Composite layout: workspace panel is at (ONE_PANEL_MARGIN, BANNER_HEIGHT)
    def on_mouse(event, win_x, win_y, _flags, _userdata):
        nonlocal current_command
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        canvas_px = win_x - ONE_PANEL_MARGIN
        canvas_py = win_y - BANNER_HEIGHT
        if not (0 <= canvas_px < workspace_size and 0 <= canvas_py < workspace_size):
            return
        # Canvas pixel -> world (same mapping as in main's visualizer)
        x_world = workspace.x_ref + (canvas_px - center) / scale
        y_world = workspace.y_ref - (canvas_py - center) / scale

        if not is_inside_workspace(x_world, y_world, workspace):
            x_world, y_world = clamp_to_workspace(x_world, y_world, workspace)
            print(f"Point outside workspace; clamped to ({x_world:.4f}, {y_world:.4f}) m")
        else:
            print(f"Moving to ({x_world:.4f}, {y_world:.4f}) m")

        cmd = TableCommand(x_des=x_world, y_des=y_world)
        actuator.send(cmd)
        current_command = cmd

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    if servo_port is not None and skip_calibration:
        print("Calibration skipped (--skip-calibration).")
    print("Click in the Workspace window to send table command. Press 'q' to quit.")
    title = "Workspace picker - click to move table | Q: quit"
    while True:
        canvas, _, _ = _render_workspace_canvas(
            workspace, current_command, workspace_size=workspace_size, grid_step_m=grid_step_m
        )
        composite = build_composite(title, None, None, canvas)
        cv2.imshow(WINDOW_NAME, composite)
        if _window_closed(WINDOW_NAME):
            break
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Pick a point in the workspace; send TableCommand to servos and visualize."
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Serial port for servos (e.g. /dev/ttyUSB1). Use 'None' for mock.",
    )
    parser.add_argument(
        "--workspace-radius",
        type=float,
        default=DEFAULT_WORKSPACE.safe_radius,
        help="Workspace safe_radius in m (default: from main).",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip terminal arrow-key workspace offset calibration (real servo only).",
    )
    args = parser.parse_args()
    port = None if args.port.strip().lower() == "none" else args.port
    workspace = WorkspaceParams(
        x_ref=DEFAULT_WORKSPACE.x_ref,
        y_ref=DEFAULT_WORKSPACE.y_ref,
        safe_radius=args.workspace_radius,
    )
    run(workspace, DEFAULT_MECHANISM, port, skip_calibration=args.skip_calibration)


if __name__ == "__main__":
    main()
