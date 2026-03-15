---
name: DVS calibration
overview: "grid-sweep total-system calibration in hardware/dvs_camera_calibration.py: configurable grid of points inside the workspace circle, step-wait-record per point, experiment-style camera views (line only), workspace view with grid and green/red/blue point states, terminal logging of saved values, and 2D interpolation at runtime."
todos: []
isProject: false
---

# Grid-sweep DVS calibration


**Pipeline order**

1. **Servo calibration first** (Arduino). Then run the calibration tool.
2. **Calibration tool** ([hardware/dvs_camera_calibration.py](hardware/dvs_camera_calibration.py)): builds a grid of points inside the workspace, runs **step → wait → record** for each point, saves (command, observation) to JSON.
3. **Runtime**: Load calibration; apply 2D interpolation (b1, b2) → (X, Y).

**Grid definition**

- Workspace: circle with center (x_ref, y_ref) and radius safe_radius (from params or CLI).
- **All calibration points must lie inside this circle.** Generate a grid (e.g. rectangular) and **discard or skip** any point whose distance from (x_ref, y_ref) exceeds safe_radius.
- **Configurable density**: user chooses how many points (e.g. `--grid 5` for 5×5, or `--grid-n 25` for 25 points). Logic: e.g. Nx × Ny with Nx, Ny chosen so total ≤ N and points span the circle; or generate points on a regular grid and keep only those inside the circle. Exact rule: grid axis-aligned, step size or count chosen so coverage is inside the circle.

**Step–wait–record loop**

- For each grid point (x_i, y_i) in a fixed order (e.g. row-major):
  1. **Step**: Send `TableCommand(x_ref + x_i, y_ref + y_i)` to the actuator (repeatedly at actuator rate for one "step" or for a fixed duration so the table moves there).
  2. **Wait**: Wait a settle time (e.g. 1–2 s or configurable) so the table is stable.
  3. **Record**: Read current line from both cameras (Hough/Sam algo output); convert to normalized (b1, b2); store (x_i, y_i, b1, b2) as one calibration sample. Optionally store (X_obs, Y_obs) from reconstruct for convenience; runtime will use (b1, b2) for interpolation.
- No manual "place pencil" — assumes calibration pencil (e.g. LED pencil) is fixed on the table so the line is visible at every position.

**Visualization (same look as experiments, except workspace)**

- **Camera visualizers (2)**: Identical to experiment view — event surfaces (from DVS) plus the **current fitted line** only (green line from Hough/Sam). **No vertical calibration reference lines**; just the single line fit like in [benchmarks/visualize_dvs_cams.py](benchmarks/visualize_dvs_cams.py) / normal run.
- **Workspace visualizer (calibration-specific)**:
  - Draw workspace circle (x_ref, y_ref, safe_radius), grid lines, center cross as in [visualization/realtime_visualizer.py](visualization/realtime_visualizer.py) `_render_workspace`.
  - Draw the **calibration grid points**:
    - **Red**: points not yet saved (pending).
    - **Green**: points already saved (we have (x, y, b1, b2) for them).
    - **Blue**: the **current table command** (the single point where the table is being sent in this step — i.e. the current (x_i, y_i) in the step–wait–record loop).
  - So at any time the user sees: full grid of dots (red/green) and one blue dot for current command. Layout and scaling same as experiment workspace (same _workspace_size, _scale, _center from workspace params).
- **Terminal**: Print clearly that we are in **calibration mode** (e.g. at start: "Calibration mode — grid sweep. Points inside workspace: N. Press Q to abort."). For each recorded point, **print the values being saved**: e.g. point index, (x_cmd, y_cmd), (b1, b2) or (X_obs, Y_obs). Optionally progress like "Point 12/25 recorded."

**Implementation notes**

- Repurpose [hardware/dvs_camera_calibration.py](hardware/dvs_camera_calibration.py): remove focus-only preview. Implement:
  - Device discovery, DVS readers, Hough or Sam (same as [benchmarks/visualize_dvs_cams.py](benchmarks/visualize_dvs_cams.py)), mechanism + actuator (same as main: [system_builder](system_builder.py), [hardware/Servo_System.py](hardware/Servo_System.py)), workspace params (x_ref, y_ref, safe_radius).
  - Build list of grid points inside the circle; sort into sweep order.
  - Main loop: for each point, send command, wait, read algo output (pixel → normalized), append to calibration list, update "saved" set for visualizer. Render at each step: cameras (event + line only), workspace (grid + green/red/blue). Check for Q to abort.
- **Workspace calibration view**: Either (a) add a dedicated calibration mode to [DVSWorkspaceVisualizer](visualization/realtime_visualizer.py) that accepts grid points + saved set + current command and draws red/green/blue, or (b) implement the same drawing logic inside the calibration script (so the script has its own small workspace canvas and draw routine). Prefer (b) to avoid coupling the main visualizer to calibration; the script can reuse the same scaling/geometry from WorkspaceParams and draw circles/lines/dots with cv2. Camera windows are the same as visualize_dvs_cams (no new visualizer class needed for cams).

**JSON output**

- Save list of points: each entry `{"x": x_cmd, "y": y_cmd, "b1": ..., "b2": ...}` (normalized). Include metadata: x_ref, y_ref, safe_radius, grid shape or count. Runtime loads and builds 2D interpolator (e.g. scipy RegularGridInterpolator from scattered points, or fit to grid). [perception/calibration.py](perception/calibration.py) (new): `DVSGridCalibration.load(path)`, `apply(b1, b2) -> (X, Y)`.

**Runtime use of calibration**

- Unchanged: [system_builder](system_builder.py) or vision layer loads calibration when path set; after get_observation + reconstruct, replace (X, Y) with calibrated (X_cal, Y_cal) from `DVSGridCalibration.apply(b1, b2)`. Angles from existing reconstruct. Visualizer in normal runs unchanged.

---

## Part 3: File changes summary


| Area          | File(s)                                                                      | Change                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pause         | [simulation/simulation_runner.py](simulation/simulation_runner.py)           | Paused state; when paused send center to actuator; pass paused to render; handle (quit, toggle_pause).                                                                                                                                                                                                                                                                                          |
| Pause         | [visualization/realtime_visualizer.py](visualization/realtime_visualizer.py) | render(..., paused); return (quit, toggle_pause); when paused draw "System paused — table at center".                                                                                                                                                                                                                                                                                           |
| Calib tool    | [hardware/dvs_camera_calibration.py](hardware/dvs_camera_calibration.py)     | Replace focus preview. Grid of points inside (x_ref, y_ref, safe_radius); step–wait–record; DVS + Hough/Sam + mechanism + actuator; **camera views**: event + current line only (no vertical ref lines); **workspace view**: grid points red (pending) / green (saved) / blue (current command); **terminal**: "Calibration mode", print each saved (x, y, b1, b2). Save JSON (list of points). |
| Calib runtime | New `perception/calibration.py`                                              | DVSGridCalibration load JSON (scattered or grid), apply(b1, b2) -> (X, Y) via 2D interpolation.                                                                                                                                                                                                                                                                                                 |
| Calib runtime | [system_builder.py](system_builder.py) / vision                              | Load calibration if path set; apply after reconstruct.                                                                                                                                                                                                                                                                                                                                          |
| Config        | [core/sim_types.py](core/sim_types.py) (optional)                            | dvs_calibration_path.                                                                                                                                                                                                                                                                                                                                                                           |


---

## Part 4: Optional / doc

- **Doc**: [docs/architecture.MD](docs/architecture.MD) — Pause (Space), calibration pipeline (servo first, then grid-sweep calibration tool), JSON format.
- **Grid size**: CLI e.g. `--grid 7` for 7×7 or `--grid-n 49`; ensure all points inside workspace circle.
- **Settle time**: Configurable wait (e.g. 1.5 s) between step and record so table is stable; optional `--settle` flag.

No change to controller or plant; pause and calibration are additive.