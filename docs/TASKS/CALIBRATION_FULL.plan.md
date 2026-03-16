---
name: CALIBRATION_FULL
overview: "Full DVS calibration pipeline: grid-sweep calibration tool (hardware/dvs_camera_calibration.py) with configurable grid, step-wait-record per point, camera/workspace visualization, JSON output; runtime gated by use_dvs_calibration with CalibratedReconstructor in vision.py (X,Y from 2D interpolation, angles from normal equations); terminal-first messaging; clear startup error when calibration requested but no JSON."
todos: []
isProject: false
---

# CALIBRATION_FULL — Complete DVS calibration plan

This document consolidates the grid-sweep calibration design and the runtime/vision update into one plan.

---

## 1. Pipeline order

1. **Servo calibration first** (Arduino). Then run the calibration tool.
2. **Calibration tool** ([hardware/dvs_camera_calibration.py](hardware/dvs_camera_calibration.py)): builds a grid of points inside the workspace, runs **step → wait → record** for each point, saves (command, observation) to JSON.
3. **Runtime**: Gated by **use_dvs_calibration**. When `use_dvs_calibration=True`, a valid calibration JSON is required (raise at startup with clear error otherwise). Load calibration; apply 2D interpolation (b1, b2) → (X, Y) via **CalibratedReconstructor**; angles from normal equations.

---

## 2. Parameter and calibration path

- **Config:** Add `use_dvs_calibration: bool = False` and `dvs_calibration_path: str | None = None` (e.g. on [HardwareParams](core/sim_types.py)).
- **When `use_dvs_calibration=True`:**
  - Path must be set. If **path is a file**, use that JSON. If **path is a directory**, use the **most recent** `.json` in that directory (by mtime).
  - If path is unset, or file/dir missing or contains no JSON, **raise at startup** with a clear message: user must run the calibration tool first and provide a valid path.
- **When `use_dvs_calibration=False`:** Pass `None` as the calibration object to vision; no loading, no error.

---

## 3. Grid definition

- Workspace: circle with center (x_ref, y_ref) and radius safe_radius (from params or CLI).
- **All calibration points must lie inside this circle.** Generate a grid (e.g. rectangular) and **discard or skip** any point whose distance from (x_ref, y_ref) exceeds safe_radius.
- **Configurable density**: user chooses how many points (e.g. `--grid 5` for 5×5, or `--grid-n 25` for 25 points). Logic: e.g. Nx × Ny with Nx, Ny chosen so total ≤ N and points span the circle; or generate points on a regular grid and keep only those inside the circle. Exact rule: grid axis-aligned, step size or count chosen so coverage is inside the circle.

---

## 4. Step–wait–record loop

- For each grid point (x_i, y_i) in a fixed order (e.g. row-major):
  1. **Step**: Send `TableCommand(x_ref + x_i, y_ref + y_i)` to the actuator (repeatedly at actuator rate for one "step" or for a fixed duration so the table moves there).
  2. **Wait**: Wait a settle time (e.g. 1–2 s or configurable) so the table is stable.
  3. **Record**: Read current line from both cameras (Hough/Sam algo output); convert to normalized (b1, b2); store (x_i, y_i, b1, b2) as one calibration sample. Optionally store (X_obs, Y_obs) from reconstruct for convenience; runtime will use (b1, b2) for interpolation.
- No manual "place pencil" — assumes calibration pencil (e.g. LED pencil) is fixed on the table so the line is visible at every position.

---

## 5. Visualization (same look as experiments, except workspace)

- **Camera visualizers (2)**: Identical to experiment view — event surfaces (from DVS) plus the **current fitted line** only (green line from Hough/Sam). **No vertical calibration reference lines**; just the single line fit like in [hardware/visualize_dvs_cams.py](hardware/visualize_dvs_cams.py) / normal run.
- **Workspace visualizer (calibration-specific)**:
  - Draw workspace circle (x_ref, y_ref, safe_radius), grid lines, center cross as in [visualization/realtime_visualizer.py](visualization/realtime_visualizer.py) `_render_workspace`.
  - Draw the **calibration grid points**:
    - **Red**: points not yet saved (pending).
    - **Green**: points already saved (we have (x, y, b1, b2) for them).
    - **Blue**: the **current table command** (the single point where the table is being sent in this step — i.e. the current (x_i, y_i) in the step–wait–record loop).
  - So at any time the user sees: full grid of dots (red/green) and one blue dot for current command. Layout and scaling same as experiment workspace (same _workspace_size, _scale, _center from workspace params).
- **Terminal**: Print clearly that we are in **calibration mode** (e.g. at start: "Calibration mode — grid sweep. Points inside workspace: N. Press Q to abort."). For each recorded point, **print the values being saved**: e.g. point index, (x_cmd, y_cmd), (b1, b2) or (X_obs, Y_obs). Optionally progress like "Point 12/25 recorded."

---

## 6. CalibratedReconstructor (vision.py)

- **Location:** [perception/vision.py](perception/vision.py).
- **Name:** `CalibratedReconstructor`.
- **Role:** Holds calibration points and performs full reconstruction from `CameraPair` to `PoseMeasurement` when calibration is used.

**Attributes:**

- Calibration points (and whatever is needed for 2D interpolation, e.g. an interpolator built from points or the raw list).

**Constructor:**

- Accepts calibration data (e.g. list of `{x, y, b1, b2}`) and camera params (xr, yr for angle formulas). Builds the (b1, b2) → (X, Y) interpolator internally. Does not read paths or "most recent" logic; loading is done in system_builder (or a small loader module).

**Single public method:**

- `reconstruct(self, cams: CameraPair) -> PoseMeasurement`
  - Get (b1, s1, b2, s2) from `cams` (same as `get_measurements`).
  - **(X, Y)** = interpolate from calibration points using (b1, b2).
  - **Angles** = same formulas as current [vision.reconstruct](perception/vision.py) (no interpolation): `denom = b1*b2+1`, then `alpha_x = (s1 + b1*s2)/denom`, `alpha_y = (s2 - b2*s1)/denom`.
  - Return `PoseMeasurement(X, Y, alpha_x, alpha_y)`.

---

## 7. Where calibration is applied

- Only inside **reconstruct**: if `self.calibration is not None`, return `self.calibration.reconstruct(cams)`. Otherwise keep current implementation (existing X, Y, alpha_x, alpha_y formulas).
- [system_builder](system_builder.py) or vision layer loads calibration when path set; after get_observation + reconstruct, (X, Y) come from calibrated reconstruction when calibration is enabled. Angles from existing reconstruct formulas (via CalibratedReconstructor). Visualizer in normal runs unchanged.

---

## 8. Wiring: system_builder and vision

- **system_builder:** When building vision for DVS:
  - If `use_dvs_calibration=True`: resolve path (file or latest JSON in dir), load calibration points, instantiate `CalibratedReconstructor` with that data (and camera_params for angle math), pass the instance into the vision constructor.
  - If `use_dvs_calibration=False`: pass `calibration=None`.
  - If calibration is requested but no valid JSON is found, raise before creating vision (clear error: must calibrate first).
- **Vision model** ([VisionModelBase](perception/vision.py) or the concrete class that owns `reconstruct`): Constructor accepts `calibration: CalibratedReconstructor | None = None` and stores it (e.g. `self.calibration`). Subclasses (RealEventCameraInterface, SimEventCameraInterface, SimVisionModel) must accept and forward this argument.

No other modules need to change for this wiring; only system_builder and vision.py.

---

## 9. Loading calibration data

- **Option A (preferred):** Small loader in [perception/calibration.py](perception/calibration.py): e.g. `load_calibration_points(path) -> list[dict]` that reads JSON and returns the list of points; system_builder calls it, then builds `CalibratedReconstructor(points, camera_params)`. The class in vision.py stays independent of path resolution.

---

## 10. JSON output (calibration tool)

- Save list of points: each entry `{"x": x_cmd, "y": y_cmd, "b1": ..., "b2": ...}` (normalized). Include metadata: x_ref, y_ref, safe_radius, grid shape or count. Runtime loads and builds 2D interpolator (e.g. scipy RegularGridInterpolator from scattered points, or fit to grid). [perception/calibration.py](perception/calibration.py): loader for system_builder; interpolator used inside CalibratedReconstructor in vision.py: `apply(b1, b2) -> (X, Y)` conceptually (inside `reconstruct`).

---

## 11. Terminal-first messaging

- **Runtime:** On startup, when calibration is loaded, print once to terminal (e.g. "DVS calibration loaded from &lt;path&gt; (N points)."). When calibration is requested but loading fails, the raised error message is the primary user-facing output (terminal).
- **Calibration tool:** All calibration-related messages (e.g. "Calibration mode", "Points inside workspace: N", which point is being chosen, "Point 12/25 recorded") go to the **terminal first**. Any UI equivalent is secondary.

---

## 12. Implementation notes (calibration tool)

- Repurpose [hardware/dvs_camera_calibration.py](hardware/dvs_camera_calibration.py): remove focus-only preview. Implement:
  - Device discovery, DVS readers, Hough or Sam (same as [benchmarks/visualize_dvs_cams.py](benchmarks/visualize_dvs_cams.py)), mechanism + actuator (same as main: [system_builder](system_builder.py), [hardware/Servo_System.py](hardware/Servo_System.py)), workspace params (x_ref, y_ref, safe_radius).
  - Build list of grid points inside the circle; sort into sweep order.
  - Main loop: for each point, send command, wait, read algo output (pixel → normalized), append to calibration list, update "saved" set for visualizer. Render at each step: cameras (event + line only), workspace (grid + green/red/blue). Check for Q to abort.
- **Workspace calibration view**: Either (a) add a dedicated calibration mode to [DVSWorkspaceVisualizer](visualization/realtime_visualizer.py) that accepts grid points + saved set + current command and draws red/green/blue, or (b) implement the same drawing logic inside the calibration script (so the script has its own small workspace canvas and draw routine). **Prefer (b)** to avoid coupling the main visualizer to calibration; the script can reuse the same scaling/geometry from WorkspaceParams and draw circles/lines/dots with cv2. Camera windows are the same as visualize_dvs_cams (no new visualizer class needed for cams).

---

## 13. File changes summary

| Area            | File(s)                                                                      | Change                                                                                                                                                                                                                                                                                                                                                                                          |
| --------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pause           | [simulation/simulation_runner.py](simulation/simulation_runner.py)           | Paused state; when paused send center to actuator; pass paused to render; handle (quit, toggle_pause).                                                                                                                                                                                                                                                                                          |
| Pause           | [visualization/realtime_visualizer.py](visualization/realtime_visualizer.py) | render(..., paused); return (quit, toggle_pause); when paused draw "System paused — table at center".                                                                                                                                                                                                                                                                                           |
| Config          | [core/sim_types.py](core/sim_types.py)                                       | Add `use_dvs_calibration: bool = False`, `dvs_calibration_path: str \| None = None` (e.g. HardwareParams).                                                                                                                                                                                                                                                                                        |
| Calib tool      | [hardware/dvs_camera_calibration.py](hardware/dvs_camera_calibration.py)     | Replace focus preview. Grid of points inside (x_ref, y_ref, safe_radius); step–wait–record; DVS + Hough/Sam + mechanism + actuator; **camera views**: event + current line only (no vertical ref lines); **workspace view**: grid points red (pending) / green (saved) / blue (current command); **terminal**: "Calibration mode", print each saved (x, y, b1, b2). Save JSON (list of points). |
| Calib runtime   | [perception/calibration.py](perception/calibration.py) (new or extend)       | `load_calibration_points(path)` returns list of points for system_builder; path can be file or directory (most recent .json). Optionally DVSGridCalibration/helpers for interpolator if kept separate from vision.                                                                                                                                                                                 |
| Calib runtime   | [perception/vision.py](perception/vision.py)                                 | **CalibratedReconstructor**: holds points + interpolator; `reconstruct(cams) -> PoseMeasurement` (X,Y from interpolation, angles from normal equations). Vision model constructor accepts `calibration: CalibratedReconstructor \| None = None`; `reconstruct` delegates to `self.calibration.reconstruct(cams)` when not None, else current equations.                                                |
| Calib runtime   | [system_builder.py](system_builder.py)                                       | When `use_dvs_calibration=True`: resolve path (file or latest JSON in dir), load points via calibration loader, create CalibratedReconstructor, pass to vision. When False: pass None. Raise clear error if calibration requested but no valid JSON.                                                                                                                                           |
| Optional doc    | [docs/architecture.MD](docs/architecture.MD)                                | Pause (Space), calibration pipeline (servo first, then grid-sweep calibration tool), JSON format, use_dvs_calibration and path, terminal-first messaging.                                                                                                                                                                                                                                       |

---

## 14. Optional / doc

- **Doc**: [docs/architecture.MD](docs/architecture.MD) — Pause (Space), calibration pipeline (servo first, then grid-sweep calibration tool), JSON format, use_dvs_calibration, dvs_calibration_path, terminal-first messaging.
- **Grid size**: CLI e.g. `--grid 7` for 7×7 or `--grid-n 49`; ensure all points inside workspace circle.
- **Settle time**: Configurable wait (e.g. 1.5 s) between step and record so table is stable; optional `--settle` flag.

No change to controller or plant; pause and calibration are additive.

---

## Summary table

| Topic    | Detail                                                                                                    |
| -------- | --------------------------------------------------------------------------------------------------------- |
| Param    | `use_dvs_calibration` (bool), `dvs_calibration_path` (str \| None)                                        |
| Path     | File → use that JSON; directory → most recent .json by mtime; else raise at startup                        |
| Class    | `CalibratedReconstructor` in vision.py; points + interpolator; `reconstruct(cams) -> PoseMeasurement`   |
| X,Y      | From 2D interpolation over calibration points                                                             |
| Angles   | Normal equations (same as current reconstruct), no interpolation                                         |
| Wiring   | system_builder creates CalibratedReconstructor or passes None; vision.reconstruct branches on calibration |
| Error    | When calibration=True and no valid JSON, raise at startup                                                |
| Messages | Terminal first (runtime: loaded / error; calibration tool: mode, points, progress)                        |
