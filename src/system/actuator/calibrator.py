"""
TPS actuator calibrator — interactive dense-grid servo calibration.

For each grid point the servo auto-moves to the nominal FK/IK position; the
operator WASD-nudges until the DVS reads the desired position, then presses
Enter to record (desired_xy, servo_xy, θ₁, θ₄, actual_xy from DVS).

Usage:
    python -m src.system.actuator.calibrator
    python -m src.system.actuator.calibrator --grid-step 0.02
    python -m src.system.actuator.calibrator --grid-step 0.005 --workspace-radius 0.06
"""

from __future__ import annotations

import argparse
import curses
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import serial

from src.shared import ControlInput, build_from_registry
from src.system.actuator.mech import Mechanism, MechanismParams, MECHANISM_PRESETS
from src.system.sensor.interface import VISION_INTERFACE_REGISTRY


# ── Grid ──────────────────────────────────────────────────────────────────────

def generate_grid(radius_m: float, step_m: float) -> list[np.ndarray]:
    """All (x, y) inside radius_m on a step_m grid, snake-scan order."""
    vals = np.arange(-radius_m, radius_m + step_m * 0.5, step_m)
    pts = [
        np.array([x, y])
        for x in vals
        for y in vals
        if np.hypot(x, y) <= radius_m
    ]
    pts.sort(key=lambda p: (round(p[1] / step_m), round(p[0] / step_m)))
    return pts


# ── Serial helpers ─────────────────────────────────────────────────────────────

def _serial_send(ser: serial.Serial, cmd: str) -> None:
    ser.write((cmd.strip() + "\r\n").encode("utf-8"))
    ser.flush()


def _send_angles(ser: serial.Serial, theta1_deg: float, theta4_deg: float) -> None:
    _serial_send(ser, f"CMD,{theta1_deg:.2f},{theta4_deg:.2f}")


def _get_angles(
    mechanism: Mechanism, servo_xy: np.ndarray
) -> tuple[float, float]:
    """FK/IK for current servo_xy (calibration disabled on mechanism)."""
    _, angles = mechanism.command_geometry(ControlInput(float(servo_xy[0]), float(servo_xy[1])))
    return angles  # (theta1_deg, theta4_deg)


# ── Smooth move ────────────────────────────────────────────────────────────────

def _smooth_move(
    mechanism: Mechanism,
    ser: serial.Serial,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    duration_s: float = 0.4,
) -> None:
    start_xy = np.asarray(start_xy, dtype=float)
    end_xy   = np.asarray(end_xy,   dtype=float)
    if np.allclose(start_xy, end_xy):
        t1, t4 = _get_angles(mechanism, end_xy)
        _send_angles(ser, t1, t4)
        return
    n = max(int(np.ceil(duration_s / 0.02)), 1)
    for i in range(1, n + 1):
        alpha = i / n
        xy = (1.0 - alpha) * start_xy + alpha * end_xy
        t1, t4 = _get_angles(mechanism, xy)
        _send_angles(ser, t1, t4)
        if i < n:
            time.sleep(0.02)


# ── DVS polling ───────────────────────────────────────────────────────────────

def _read_dvs(sensor) -> tuple[float, float] | None:
    try:
        y = sensor.get_y(None)
        return float(y.px), float(y.py)
    except Exception:
        return None


# ── Warm-start seeds ──────────────────────────────────────────────────────────

def _load_existing(output_path: Path) -> dict[str, dict]:
    """Return dict keyed by 'x,y' rounded string → saved record."""
    if not output_path.exists():
        return {}
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
        if data.get("version") != "v1_tps_actuator":
            return {}
        result = {}
        for rec in data["points"]:
            key = _xy_key(np.array(rec["desired_xy_m"]))
            result[key] = rec
        return result
    except Exception:
        return {}


def _xy_key(xy: np.ndarray) -> str:
    return f"{xy[0]:.6f},{xy[1]:.6f}"


# ── Curses UI ─────────────────────────────────────────────────────────────────

def _draw(stdscr, row: int, text: str) -> None:
    h, w = stdscr.getmaxyx()
    if row < h:
        stdscr.addstr(row, 0, text[: max(0, w - 1)])


def _calibration_loop(
    stdscr,
    *,
    mechanism: Mechanism,
    ser: serial.Serial,
    sensor,
    grid_points: list[np.ndarray],
    nudge_step_m: float,
    existing: dict[str, dict],
    args,
) -> list[dict]:
    curses.curs_set(0)
    stdscr.keypad(True)
    stdscr.timeout(50)

    accepted: list[dict] = []
    last_dvs: tuple[float, float] | None = None
    last_dvs_t = 0.0
    last_action = "Starting calibration."
    current_xy = np.zeros(2)

    total = len(grid_points)
    idx = 0

    while idx < total:
        desired_xy = grid_points[idx]
        key_str = _xy_key(desired_xy)

        saved = existing.get(key_str)
        if saved is not None:
            seed_xy = np.array(saved["servo_xy_m"], dtype=float)
            seed_src = "saved"
        else:
            seed_xy = desired_xy.copy()
            seed_src = "FK/IK nominal"

        servo_xy = seed_xy.copy()

        _smooth_move(mechanism, ser, current_xy, servo_xy)
        current_xy = servo_xy.copy()
        last_action = f"Moved to {seed_src} seed."

        while True:
            now = time.time()

            # Poll DVS at ~12 Hz
            if now - last_dvs_t > 0.08:
                reading = _read_dvs(sensor)
                if reading is not None:
                    last_dvs = reading
                last_dvs_t = now

            key = stdscr.getch()
            dx = dy = 0.0

            if key != -1:
                if key in (ord("w"), ord("W"), curses.KEY_UP):
                    dy = +nudge_step_m
                    last_action = "Nudged +Y"
                elif key in (ord("s"), ord("S"), curses.KEY_DOWN):
                    dy = -nudge_step_m
                    last_action = "Nudged -Y"
                elif key in (ord("a"), ord("A"), curses.KEY_LEFT):
                    dx = -nudge_step_m
                    last_action = "Nudged -X"
                elif key in (ord("d"), ord("D"), curses.KEY_RIGHT):
                    dx = +nudge_step_m
                    last_action = "Nudged +X"
                elif key in (ord("r"), ord("R")):
                    servo_xy = seed_xy.copy()
                    _smooth_move(mechanism, ser, current_xy, servo_xy, duration_s=0.2)
                    current_xy = servo_xy.copy()
                    last_action = "Reset to seed."
                elif key in (ord("b"), ord("B")):
                    if idx > 0:
                        idx -= 1
                    last_action = "Back to previous point."
                    break
                elif key in (ord(" "),):
                    # Skip: record with whatever servo_xy/DVS we have now
                    t1, t4 = _get_angles(mechanism, servo_xy)
                    accepted.append({
                        "desired_xy_m": desired_xy.tolist(),
                        "servo_xy_m":   servo_xy.tolist(),
                        "theta1_deg":   t1,
                        "theta4_deg":   t4,
                        "actual_xy_m":  list(last_dvs) if last_dvs else None,
                        "skipped":      True,
                    })
                    idx += 1
                    last_action = "Skipped (recorded as-is)."
                    break
                elif key in (10, 13, curses.KEY_ENTER):
                    t1, t4 = _get_angles(mechanism, servo_xy)
                    accepted.append({
                        "desired_xy_m": desired_xy.tolist(),
                        "servo_xy_m":   servo_xy.tolist(),
                        "theta1_deg":   t1,
                        "theta4_deg":   t4,
                        "actual_xy_m":  list(last_dvs) if last_dvs else None,
                        "skipped":      False,
                    })
                    idx += 1
                    last_action = "Accepted."
                    break
                elif key in (ord("q"), ord("Q")):
                    return accepted  # save what we have

                if dx != 0.0 or dy != 0.0:
                    servo_xy = servo_xy + np.array([dx, dy])
                    t1, t4 = _get_angles(mechanism, servo_xy)
                    _send_angles(ser, t1, t4)
                    current_xy = servo_xy.copy()

            # ── Render ───────────────────────────────────────────────────────
            t1, t4 = _get_angles(mechanism, servo_xy)
            stdscr.erase()
            _draw(stdscr, 0, f"TPS Actuator Calibrator — point {idx + 1}/{total}")
            _draw(stdscr, 1, "WASD/arrows: nudge  Enter: accept  Space: skip  R: reset  B: back  Q: quit+save")
            _draw(stdscr, 2, f"Grid step: {args.grid_step * 1000:.0f} mm   Nudge step: {nudge_step_m * 1000:.1f} mm   Radius: {args.workspace_radius * 100:.1f} cm")
            _draw(stdscr, 4, f"Target:  ({desired_xy[0]:+.4f}, {desired_xy[1]:+.4f}) m")
            _draw(stdscr, 5, f"Command: ({servo_xy[0]:+.4f}, {servo_xy[1]:+.4f}) m   θ₁={t1:.2f}°  θ₄={t4:.2f}°")
            if last_dvs is not None:
                ex = last_dvs[0] - desired_xy[0]
                ey = last_dvs[1] - desired_xy[1]
                _draw(stdscr, 6, f"DVS:     ({last_dvs[0]:+.4f}, {last_dvs[1]:+.4f}) m")
                _draw(stdscr, 7, f"Error:   ({ex:+.4f}, {ey:+.4f}) m   |err|={np.hypot(ex, ey)*1000:.2f} mm")
            else:
                _draw(stdscr, 6, "DVS:     waiting for observation...")
            _draw(stdscr, 9,  f"Seed source: {seed_src}")
            _draw(stdscr, 10, f"Accepted so far: {len(accepted)}/{total}")
            _draw(stdscr, 11, f"Last action: {last_action}")
            stdscr.refresh()

    return accepted


# ── Save ──────────────────────────────────────────────────────────────────────

def _save(output_path: Path, accepted: list[dict], args) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version":           "v1_tps_actuator",
        "updated_at_utc":    datetime.now(timezone.utc).isoformat(),
        "grid_step_m":       args.grid_step,
        "workspace_radius_m": args.workspace_radius,
        "n_points":          len(accepted),
        "points":            accepted,
        "metadata": {
            "port":  args.port,
            "tool":  "src.system.actuator.calibrator",
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(accepted)} points → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive TPS actuator calibration — WASD dense-grid tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--grid-step",         type=float, default=0.01,
                        help="grid spacing in meters (default: 0.01 = 10 mm)")
    parser.add_argument("--workspace-radius",  type=float, default=0.07,
                        help="calibration radius in meters (default: 0.07 = 7 cm)")
    parser.add_argument("--nudge-step",        type=float, default=0.001,
                        help="WASD nudge size in meters (default: 0.001 = 1 mm)")
    parser.add_argument("--port",              default="/dev/ttyUSB0",
                        help="serial port (default: /dev/ttyUSB0)")
    parser.add_argument("--baud",              type=int, default=115200,
                        help="baud rate (default: 115200)")
    parser.add_argument("--output",            default="src/system/actuator/mech/five_bar_tps_calibration.json",
                        help="output JSON file path")
    parser.add_argument("--sensor-preset",     default="real_dvs:hough",
                        help="DVS sensor preset (default: real_dvs:hough)")
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    # Build mechanism (FK/IK seed only; calibration disabled)
    mech_params = MechanismParams(**{
        k: v for k, v in MECHANISM_PRESETS["default"].items()
        if k != "calibration_file"
    }, calibration_file=None)
    mechanism = Mechanism(mech_params)
    mechanism.set_calibration_enabled(False)
    mechanism.set_workspace_offset(0.0, 0.0)

    # Open serial
    print(f"Opening {args.port} at {args.baud} baud…")
    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(2)
    _serial_send(ser, "MODE,EXP")
    print("Serial ready.")

    # Build DVS sensor
    print(f"Starting DVS sensor ({args.sensor_preset})…")
    sensor = build_from_registry(VISION_INTERFACE_REGISTRY, args.sensor_preset)
    time.sleep(1.0)  # let background threads warm up
    print("DVS ready.")

    # Grid and warm-start data
    grid_points = generate_grid(args.workspace_radius, args.grid_step)
    existing    = _load_existing(output_path)
    print(f"Grid: {len(grid_points)} points  |  Warm-start hits: {len(existing)}")

    try:
        accepted = curses.wrapper(
            lambda stdscr: _calibration_loop(
                stdscr,
                mechanism=mechanism,
                ser=ser,
                sensor=sensor,
                grid_points=grid_points,
                nudge_step_m=args.nudge_step,
                existing=existing,
                args=args,
            )
        )
    except KeyboardInterrupt:
        accepted = []
    finally:
        ser.close()

    if accepted:
        _save(output_path, accepted, args)
    else:
        print("No points accepted; nothing saved.")


if __name__ == "__main__":
    main()
