"""
Interactive 5-point real-servo calibration.

Each run walks the user through center + four cardinal points. For every point:
1. Move to the saved servo-space seed if one exists, otherwise the nominal point.
2. Let the user nudge the raw servo-space command with WASD / arrow keys.
3. Save the final raw servo-space command for that named point.

After all five points are accepted, an affine desired->servo map is fitted and
written to the mechanism calibration file.
"""

from __future__ import annotations

import curses
import time

import numpy as np

from src.shared import ControlInput, WorkspaceParams, Measurement


def _read_y_meas_from_vision(vision) -> Measurement | None:
    if vision is None:
        return None
    try:
        measurement = vision.get_observation(None)
        if measurement is None:
            return None
        return vision.reconstruct(measurement)
    except Exception:
        return None


def _draw_line(stdscr, row: int, text: str) -> None:
    height, width = stdscr.getmaxyx()
    if row >= height:
        return
    stdscr.addstr(row, 0, text[: max(0, width - 1)])


def _point_banner(name: str, idx: int, total: int, desired_xy: np.ndarray) -> str:
    return (
        f"Point {idx + 1}/{total}: {name.upper()} | "
        f"desired physical target=({desired_xy[0]:+.4f}, {desired_xy[1]:+.4f}) m"
    )


def _send_raw_workspace_command(actuator, servo_xy: np.ndarray) -> None:
    actuator.send(ControlInput(px_cmd=float(servo_xy[0]), py_cmd=float(servo_xy[1])))


def _move_raw_workspace_command_smooth(
    actuator,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    *,
    duration_s: float = 0.35,
) -> np.ndarray:
    start_xy = np.asarray(start_xy, dtype=float).reshape(2)
    end_xy = np.asarray(end_xy, dtype=float).reshape(2)

    if duration_s <= 0.0 or np.allclose(start_xy, end_xy):
        _send_raw_workspace_command(actuator, end_xy)
        return end_xy.copy()

    period_s = float(getattr(actuator, "period", 0.02))
    step_s = min(max(period_s, 0.01), 0.05)
    n_steps = max(int(np.ceil(duration_s / step_s)), 1)

    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        servo_xy = (1.0 - alpha) * start_xy + alpha * end_xy
        _send_raw_workspace_command(actuator, servo_xy)
        if i < n_steps:
            time.sleep(step_s)

    return end_xy.copy()


def _curses_calibration_loop(
    stdscr,
    *,
    system,
    actuator,
    workspace: WorkspaceParams,
    nudge_step_m: float,
    cardinal_delta_m: float,
    move_duration_s: float,
) -> dict[str, dict[str, np.ndarray]]:
    vision = None
    try:
        perception = getattr(system, "perception", None)
        vision = getattr(perception, "vision", None) if perception is not None else None
    except Exception:
        vision = None

    mechanism = getattr(actuator, "mechanism", None)
    if mechanism is None or not hasattr(mechanism, "calibration_targets"):
        raise TypeError("actuator.mechanism must support persisted 5-point calibration")

    if hasattr(actuator, "set_workspace_offset"):
        actuator.set_workspace_offset(0.0, 0.0)
    if hasattr(actuator, "set_calibration_enabled"):
        actuator.set_calibration_enabled(False)

    points = mechanism.calibration_targets(
        x_ref=float(workspace.x_ref),
        y_ref=float(workspace.y_ref),
        cardinal_delta_m=float(cardinal_delta_m),
        safe_radius=workspace.safe_radius,
    )

    stdscr.clear()
    curses.curs_set(0)
    stdscr.keypad(True)
    stdscr.timeout(50)

    accepted: dict[str, dict[str, np.ndarray]] = {}
    last_y_meas: Measurement | None = None
    last_y_meas_t = 0.0
    last_send_t = 0.0
    last_action = "Starting calibration."
    current_servo_xy = np.array([float(workspace.x_ref), float(workspace.y_ref)], dtype=float)

    total = len(points)
    for idx, point in enumerate(points):
        name = str(point["name"])
        desired_xy = np.asarray(point["desired_xy_m"], dtype=float).reshape(2)
        servo_xy = np.asarray(point["seed_servo_xy_m"], dtype=float).reshape(2).copy()
        saved_before = bool(mechanism.calibration_points().get(name))

        current_servo_xy = _move_raw_workspace_command_smooth(
            actuator,
            current_servo_xy,
            servo_xy,
            duration_s=move_duration_s,
        )
        last_send_t = time.time()
        last_action = "Moved to saved point." if saved_before else "Moved to nominal point."

        while True:
            now = time.time()
            if now - last_y_meas_t > 0.08:
                y_meas = _read_y_meas_from_vision(vision)
                if y_meas is not None:
                    last_y_meas = y_meas
                last_y_meas_t = now

            key = stdscr.getch()
            if key != -1:
                dx = 0.0
                dy = 0.0
                if key in (ord("w"), ord("W"), curses.KEY_UP):
                    dy = +nudge_step_m
                    last_action = "Nudged UP"
                elif key in (ord("s"), ord("S"), curses.KEY_DOWN):
                    dy = -nudge_step_m
                    last_action = "Nudged DOWN"
                elif key in (ord("a"), ord("A"), curses.KEY_LEFT):
                    dx = -nudge_step_m
                    last_action = "Nudged LEFT"
                elif key in (ord("d"), ord("D"), curses.KEY_RIGHT):
                    dx = +nudge_step_m
                    last_action = "Nudged RIGHT"
                elif key in (ord("r"), ord("R")):
                    servo_xy = np.asarray(point["seed_servo_xy_m"], dtype=float).reshape(2).copy()
                    current_servo_xy = _move_raw_workspace_command_smooth(
                        actuator,
                        current_servo_xy,
                        servo_xy,
                        duration_s=move_duration_s,
                    )
                    last_send_t = now
                    last_action = "Reset to saved seed." if saved_before else "Reset to nominal seed."
                elif key in (10, 13, curses.KEY_ENTER):
                    accepted[name] = {
                        "desired_xy_m": desired_xy.copy(),
                        "servo_xy_m": servo_xy.copy(),
                    }
                    last_action = "Accepted point."
                    break
                elif key in (ord("q"), ord("Q")):
                    raise RuntimeError("5-point servo calibration aborted by user (q).")

                if dx != 0.0 or dy != 0.0:
                    servo_xy += np.array([dx, dy], dtype=float)
                    _send_raw_workspace_command(actuator, servo_xy)
                    current_servo_xy = servo_xy.copy()
                    last_send_t = now

            _draw_line(stdscr, 0, "Pre-run 5-point servo calibration")
            _draw_line(stdscr, 1, "WASD/arrows: nudge raw servo command | Enter: accept | R: reset | Q: abort")
            _draw_line(stdscr, 2, f"Nudge step: {nudge_step_m:.4f} m | Cardinal delta: {cardinal_delta_m:.4f} m")
            _draw_line(stdscr, 4, _point_banner(name, idx, total, desired_xy))
            _draw_line(
                stdscr,
                5,
                f"Seed source: {'saved calibration' if saved_before else 'nominal target'}",
            )
            _draw_line(stdscr, 6, f"Current raw servo command=({servo_xy[0]:+.4f}, {servo_xy[1]:+.4f}) m")
            _draw_line(stdscr, 7, f"Last action: {last_action}")

            if last_y_meas is None:
                _draw_line(stdscr, 9, "Measured y_meas: waiting for vision...")
            else:
                err_xy = np.array([last_y_meas.px, last_y_meas.py], dtype=float) - desired_xy
                _draw_line(
                    stdscr,
                    9,
                    (
                        "Measured y_meas="
                        f"({last_y_meas.px:+.4f}, {last_y_meas.py:+.4f}) m | "
                        f"error vs target=({err_xy[0]:+.4f}, {err_xy[1]:+.4f}) m"
                    ),
                )

            _draw_line(
                stdscr,
                11,
                "Adjust until the end effector sits on the named physical point, then press Enter.",
            )
            _draw_line(stdscr, 13, f"Last command sent {now - last_send_t:+.2f}s ago")
            stdscr.refresh()

    return accepted


def calibrate_servo_workspace_offset(
    *,
    system,
    actuator,
    workspace: WorkspaceParams,
    step_m: float = 0.001,
    cardinal_delta_m: float = 0.015,
    move_duration_s: float = 0.35,
) -> tuple[float, float]:
    """
    Backward-compatible wrapper around the new 5-point calibration flow.

    Returns the center-point translation for older callers, but also persists the
    full five-point calibration directly into the mechanism.
    """
    if not hasattr(actuator, "send"):
        raise TypeError("actuator must provide send(ControlInput)")

    mechanism = getattr(actuator, "mechanism", None)
    if mechanism is None or not hasattr(mechanism, "save_calibration"):
        raise TypeError("actuator.mechanism must support save_calibration")

    try:
        accepted = curses.wrapper(
            lambda stdscr: _curses_calibration_loop(
                stdscr,
                system=system,
                actuator=actuator,
                workspace=workspace,
                nudge_step_m=step_m,
                cardinal_delta_m=cardinal_delta_m,
                move_duration_s=move_duration_s,
            )
        )
    except curses.error as exc:
        raise RuntimeError(
            f"Curses UI failed (terminal may be too small or unsupported): {exc}"
        ) from exc
    finally:
        if hasattr(actuator, "set_calibration_enabled"):
            actuator.set_calibration_enabled(True)

    mechanism.save_calibration(accepted)
    center = accepted["center"]
    center_delta = np.asarray(center["servo_xy_m"], dtype=float) - np.asarray(center["desired_xy_m"], dtype=float)
    print(
        "Calibration accepted and saved: "
        f"{len(accepted)} points -> {getattr(mechanism, 'calibration_path', 'configured file')}"
    )
    return float(center_delta[0]), float(center_delta[1])
