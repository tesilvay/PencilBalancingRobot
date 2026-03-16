# DVS calibration

This document describes how the current DVS (event-camera) calibration works: the **calibration tool** that records (table position, line intercepts) over a grid, the **JSON format** and **runtime class** that map intercepts to workspace (X, Y), and how it is **wired into the vision pipeline**.

## Overview

The vision pipeline turns two camera line fits into a **pose measurement**: (X, Y) base position and (alpha_x, alpha_y) tilt angles. The analytic `reconstruct()` in `perception/vision.py` computes (X, Y) from the **normalized intercepts** (b1, b2) using camera geometry (xr, yr). In practice, that model is not perfectly accurate (camera height, lens, alignment). Calibration records **ground truth**: we move the table to known (x_cmd, y_cmd) and record the (b1, b2) that the cameras actually see. At runtime we use that lookup instead of (or as a correction to) the geometric (X, Y).

**Current design:**

1. **Calibration tool** (`hardware/dvs_camera_calibration.py`): grid of points inside the workspace; for each point, send table command → wait for settle → read line fits from both cameras → save (x_cmd, y_cmd, b1, b2) to JSON.
2. **Runtime** (`perception/calibration.py`): load JSON, build a 2D interpolator (b1, b2) → (X, Y). Vision’s `reconstruct()` uses this when a calibration is attached and the interpolated (X, Y) is finite; otherwise it keeps the geometric (X, Y). Angles (alpha_x, alpha_y) always come from the existing geometric formulas (slopes s1, s2).

So calibration **replaces** the (X, Y) part of the reconstruct with a data-driven map; tilt angles are not calibrated.

---

## 1. Calibration tool (grid sweep)

**Script:** `hardware/dvs_camera_calibration.py`  
**Run:** `python -m hardware.dvs_camera_calibration` (optionally with flags).

### Prerequisites

- **Servo calibration** must be done first so table (x, y) commands correspond to real positions.
- DVS cameras and mechanism connected; pencil (e.g. LED pencil) fixed on the table so the line is visible at every grid position.

### What it does

1. **Grid:** Builds a set of points inside the workspace circle (center x_ref, y_ref, radius safe_radius). Points are in ref-centered coordinates; only points with distance ≤ safe_radius are kept. Grid density is controlled by `--grid N` (N×N) or `--grid-n N` (target total).
2. **Wait for Space:** Shows camera and workspace windows and waits for the user to press **Space** before starting the sweep. **Q** quits.
3. **Step–wait–record loop** (for each grid point in order):
   - **Step:** Send `TableCommand(x_ref + dx, y_ref + dy)` to the actuator.
   - **Wait:** Wait a settle time (default 2 s) so the table is stable.
   - **Record:** Drain DVS events, update the line algorithms (Hough or Sam), read the current fitted line from each camera, convert pixel line to **normalized** (b1, b2) via `CameraModel.pixel_to_normalized`, and append `{x_cmd, y_cmd, b1, b2}` to the calibration list.
4. **Visualization:**
   - Two camera windows: event surface + current fitted line (green); same style as the normal DVS view.
   - Workspace window: circle, grid, and dots — **red** = pending, **green** = saved, **blue** = current point. An arrow from the blue point to the next point shows the upcoming target.
5. **Save:** Writes JSON to the path given by `--output` (default `perception/calibration_files/dvs_calibration.json`). The directory is created if it does not exist. Each run **overwrites** the file unless you pass a different `--output`.

### Defaults and common options

| Option        | Default                              | Meaning                          |
|---------------|--------------------------------------|----------------------------------|
| `--grid`      | 3                                    | Grid side (e.g. 3×3 points)      |
| `--settle`    | 2.0                                  | Settle time in seconds           |
| `--output`    | perception/calibration_files/dvs_calibration.json | Output JSON path        |
| `--port`      | /dev/ttyUSB0                         | Servo serial port                 |
| `--mode`      | hough                                | Line algorithm: hough or sam     |

Other flags: `--cam1` / `--cam2` (serials; omit for discovery), `--workspace-radius`, `--noise-filter-duration` (Sam), etc.

---

## 2. JSON format

The tool writes an object with at least:

- **`points`**: array of samples. Each sample: `{"x": x_cmd, "y": y_cmd, "b1": ..., "b2": ...}` (meters and normalized intercepts).
- **`x_ref`**, **`y_ref`**, **`safe_radius`**: workspace parameters used for the run.
- **`grid_count`**: number of grid points.
- **`saved_count`**: number of samples actually recorded (can be less if a point had no valid line fit).

The runtime loader in `perception/calibration.py` accepts either:
- an object with a **`points`** key (and keeps the rest as metadata), or  
- a top-level array of point objects (metadata then empty).

---

## 3. Runtime calibration class

**Module:** `perception/calibration.py`  
**Class:** `DVSGridCalibration`

- **`load(path)`** (class method): Load JSON from `path` and return a `DVSGridCalibration` instance. Builds two `scipy.interpolate.LinearNDInterpolator` instances: (b1, b2) → x_cmd and (b1, b2) → y_cmd over the scattered calibration points.
- **`apply(b1, b2)`**: Returns `(X, Y)` in meters. If (b1, b2) is **inside** the convex hull of the calibration (b1, b2) points, returns the interpolated (X, Y). If **outside** the hull, the interpolators return `nan`; `apply` returns that, and the vision layer treats non-finite (X, Y) as “no calibration” and falls back to the geometric (X, Y).

So the effective transfer is: **normalized intercepts (b1, b2) → workspace position (X, Y)**. No dependence on slope/angle in the current calibration; angles are still from the analytic reconstruct.

---

## 4. Wiring into the vision pipeline

- **Config:** In `core/sim_types.py`, `HardwareParams` has **`dvs_calibration_path: str | None = None`**. When non-empty, it should point to a calibration JSON file (e.g. `perception/calibration_files/dvs_calibration.json`).
- **Loading:** In `system_builder.build_vision()`, when building `RealEventCameraInterface` or `SimEventCameraInterface` for DVS, if `params.hardware.dvs_calibration_path` is set, the code loads `DVSGridCalibration.load(hw.dvs_calibration_path)` and assigns it to **`vision.dvs_calibration`**. Otherwise `vision.dvs_calibration = None`.
- **Use in reconstruct:** In `perception/vision.py`, `VisionModelBase.reconstruct()`:
  1. Computes (X, Y) and (alpha_x, alpha_y) from (b1, s1, b2, s2) using the existing geometric formulas.
  2. If `self.dvs_calibration` is not None, calls `self.dvs_calibration.apply(b1, b2)`. If both returned values are finite, it **replaces** (X, Y) with the calibrated (X_cal, Y_cal).
  3. Returns a `PoseMeasurement` with that (X, Y) and the same (alpha_x, alpha_y).

So when calibration is enabled and (b1, b2) is within the calibrated range, **position** comes from the calibration map; **angles** always come from the geometric model. When calibration is disabled or (b1, b2) is outside the hull, (X, Y) is the geometric value.

---

## 5. Summary

| Component              | Role                                                                 |
|------------------------|----------------------------------------------------------------------|
| Calibration tool       | Move table to grid, record (x_cmd, y_cmd, b1, b2); save JSON.        |
| JSON                   | List of points + metadata; consumed by `DVSGridCalibration.load()`.   |
| DVSGridCalibration     | 2D interpolator (b1, b2) → (X, Y); used in `reconstruct()` when set. |
| HardwareParams         | `dvs_calibration_path`: path to JSON or None.                       |
| build_vision()          | Loads calibration and attaches it to the vision object when path set. |
| reconstruct()          | Replaces (X, Y) with `dvs_calibration.apply(b1, b2)` when finite.   |

Calibration is **position-only** and **intercept-based**: it corrects the mapping from observed (b1, b2) to workspace (X, Y). Tilt (slopes / angles) is not calibrated and continues to use the analytic formulas.
