"""
Real-servo pre-run calibration: compute a workspace translation offset.

Workflow:
1. Drive the mechanism to the configured workspace "center" (x_ref, y_ref).
2. Let the user adjust commanded workspace (x_cmd, y_cmd) via terminal arrow keys.
3. When the user confirms (Enter), compute:
     x_offset = x_cmd - workspace.x_ref
     y_offset = y_cmd - workspace.y_ref
4. The offset is applied inside `hardware/Servo_System.py` before IK for every command.
"""

from __future__ import annotations

import curses
import time
from typing import Tuple

from core.sim_types import TableCommand, WorkspaceParams, PoseMeasurement


def _clamp_to_workspace(x_des: float, y_des: float, workspace: WorkspaceParams) -> tuple[float, float]:
    """Clamp (x_des,y_des) onto the workspace circle if a safe_radius is configured."""
    if workspace.safe_radius is None:
        return x_des, y_des

    x_ref = workspace.x_ref
    y_ref = workspace.y_ref
    safe_radius = workspace.safe_radius

    dx = x_des - x_ref
    dy = y_des - y_ref
    dist = (dx * dx + dy * dy) ** 0.5
    if dist > safe_radius and dist > 0:
        scale = safe_radius / dist
        dx *= scale
        dy *= scale
        x_des = x_ref + dx
        y_des = y_ref + dy
    return x_des, y_des


def _read_pose_from_vision(vision) -> PoseMeasurement | None:
    """
    Best-effort: return PoseMeasurement from the most recent DVS observation.

    Real DVS vision ignores the state_true argument and uses background threads.
    """
    if vision is None:
        return None
    try:
        measurement = vision.get_observation(None)
        if measurement is None:
            return None
        pose = vision.reconstruct(measurement)
        return pose
    except Exception:
        # Calibration UI should not crash the process if camera momentarily fails.
        return None


def _draw_line(stdscr, row: int, text: str) -> None:
    height, width = stdscr.getmaxyx()
    if row >= height:
        return
    # Avoid curses.error on narrow terminals.
    text = text[: max(0, width - 1)]
    stdscr.addstr(row, 0, text)


def _curses_calibration_loop(
    stdscr,
    *,
    system,
    actuator,
    workspace: WorkspaceParams,
    step_m: float,
) -> Tuple[float, float]:
    vision = None
    try:
        perception = getattr(system, "perception", None)
        vision = getattr(perception, "vision", None) if perception is not None else None
    except Exception:
        vision = None

    # Ensure calibration starts with no translation applied.
    if hasattr(actuator, "set_workspace_offset"):
        actuator.set_workspace_offset(0.0, 0.0)

    x_cmd = float(workspace.x_ref)
    y_cmd = float(workspace.y_ref)

    actuator.send(TableCommand(x_des=x_cmd, y_des=y_cmd))

    stdscr.clear()
    curses.curs_set(0)
    stdscr.keypad(True)
    stdscr.timeout(50)  # ms

    last_pose: PoseMeasurement | None = None
    last_pose_t = 0.0
    last_send_t = time.time()
    last_action = "Centered at origin and sent initial command."
    last_delta = (0.0, 0.0)

    def send_command() -> None:
        nonlocal last_send_t
        actuator.send(TableCommand(x_des=x_cmd, y_des=y_cmd))
        last_send_t = time.time()

    while True:
        now = time.time()

        # Read pose at a modest rate; avoids burning CPU in the calibration loop.
        if now - last_pose_t > 0.08:
            pose = _read_pose_from_vision(vision)
            if pose is not None:
                last_pose = pose
            last_pose_t = now

        key = stdscr.getch()
        if key != -1:
            # Arrow keys: nudge commanded workspace.
            if key == curses.KEY_UP:
                y_cmd += step_m
                x_cmd, y_cmd = _clamp_to_workspace(x_cmd, y_cmd, workspace)
                last_action = "Command sent: moved UP"
                last_delta = (0.0, +step_m)
                send_command()
            elif key == curses.KEY_DOWN:
                y_cmd -= step_m
                x_cmd, y_cmd = _clamp_to_workspace(x_cmd, y_cmd, workspace)
                last_action = "Command sent: moved DOWN"
                last_delta = (0.0, -step_m)
                send_command()
            elif key == curses.KEY_LEFT:
                x_cmd -= step_m
                x_cmd, y_cmd = _clamp_to_workspace(x_cmd, y_cmd, workspace)
                last_action = "Command sent: moved LEFT"
                last_delta = (-step_m, 0.0)
                send_command()
            elif key == curses.KEY_RIGHT:
                x_cmd += step_m
                x_cmd, y_cmd = _clamp_to_workspace(x_cmd, y_cmd, workspace)
                last_action = "Command sent: moved RIGHT"
                last_delta = (+step_m, 0.0)
                send_command()
            elif key in (10, 13, curses.KEY_ENTER):
                # Confirm current command as the visual origin.
                break
            elif key in (ord("r"), ord("R")):
                x_cmd = float(workspace.x_ref)
                y_cmd = float(workspace.y_ref)
                last_action = "Command sent: RESET to origin"
                last_delta = (0.0, 0.0)
                send_command()
            elif key in (ord("q"), ord("Q")):
                raise RuntimeError("Workspace offset calibration aborted by user (q).")

        _draw_line(stdscr, 0, "Pre-run workspace offset calibration")
        _draw_line(stdscr, 1, "Arrows: move command | Enter: accept origin | r: reset | q: abort")
        _draw_line(stdscr, 2, f"Step: {step_m:.4f} m")
        _draw_line(stdscr, 3, f"{last_action} | Δ=({last_delta[0]:+.4f}, {last_delta[1]:+.4f}) m")

        x_offset_now = x_cmd - float(workspace.x_ref)
        y_offset_now = y_cmd - float(workspace.y_ref)
        _draw_line(stdscr, 4, f"Command (x_cmd, y_cmd) = ({x_cmd:+.4f}, {y_cmd:+.4f}) m")
        _draw_line(stdscr, 5, f"Current offset (x_offset, y_offset) = ({x_offset_now:+.4f}, {y_offset_now:+.4f}) m")

        if last_pose is None:
            _draw_line(stdscr, 7, "Measured pose: waiting for vision...")
        else:
            err_x = last_pose.X - workspace.x_ref
            err_y = last_pose.Y - workspace.y_ref
            _draw_line(
                stdscr,
                7,
                (
                    "Measured pose: "
                    f"({last_pose.X:+.4f}, {last_pose.Y:+.4f}) m | "
                    f"error vs origin ({err_x:+.4f}, {err_y:+.4f}) m"
                ),
            )

        _draw_line(stdscr, 9, "Use the measured pose/error (or your visual cue) to align the origin.")

        # Small footer to show we are still alive.
        _draw_line(stdscr, 11, f"Last command sent {now - last_send_t:+.2f}s ago")

        stdscr.refresh()

    x_offset = x_cmd - float(workspace.x_ref)
    y_offset = y_cmd - float(workspace.y_ref)
    return x_offset, y_offset


def calibrate_servo_workspace_offset(
    *,
    system,
    actuator,
    workspace: WorkspaceParams,
    step_m: float = 0.002,
) -> tuple[float, float]:
    """
    Run a blocking terminal calibration to compute (x_offset, y_offset).

    This should only be used in real-servo + real-DVS mode where pose measurement
    is meaningful.
    """
    if not hasattr(actuator, "send"):
        raise TypeError("actuator must provide a send(TableCommand) method")

    try:
        x_offset, y_offset = curses.wrapper(
            lambda stdscr: _curses_calibration_loop(
                stdscr,
                system=system,
                actuator=actuator,
                workspace=workspace,
                step_m=step_m,
            )
        )
        # curses.wrapper returns control to the normal terminal; safe to print.
        print(f"Calibration accepted: x_offset={x_offset:+.6f} m, y_offset={y_offset:+.6f} m")
        return x_offset, y_offset
    except curses.error as exc:
        raise RuntimeError(
            f"Curses UI failed (terminal may be too small or unsupported): {exc}"
        ) from exc

