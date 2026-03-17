# Project Architecture

This repository is a **Python reimplementation** of the pencil balancing robot from:

**J. Conradt et al., “A Pencil Balancing Robot using a Pair of AER Dynamic Vision Sensors,” IEEE, 2009**

It reproduces the end-to-end pipeline:

- Event-based vision (DVS) line tracking (per camera)
- 3D pencil reconstruction from two orthogonal views
- State estimation (position/tilt + velocities)
- Control of a 2‑DOF balancing table (simulated and optionally real hardware)

---

## High-level system (dataflow)

The runtime architecture is a perception → estimation → control loop:

1. **Plant** evolves using the previous command.
2. **Vision** produces a measurement (simulated projection/noise/delay or event-camera line tracking).
3. **Reconstruction** converts two camera lines into a 3D pose \((X, Y, \alpha_x, \alpha_y)\).
4. **Estimator** turns pose into a full 8‑state estimate (`SystemState`), adding velocities.
5. **Controller** computes a new table command (`TableCommand`).
6. Optionally:
   - **Actuator** sends the command to servos (real hardware)
   - **Visualization** renders the tracking in real time

The core loop lives in:

- `simulation/simulator.py` (`Simulator.step`)
- `simulation/simulation_runner.py` (looping, scheduling, logging, optional real-time pacing)

---

## Entrypoints & orchestration

- **`main.py`**: CLI entrypoint. Selects run mode:
  - `single` (one trial)
  - `benchmark_single` / `benchmark_all_configs`
  - `graph_results`
  - `sweep_workspace`
- **`experiments/`**: experiment runners and Monte Carlo trials.
  - `experiments/experiments.py` wires experiment modes to the simulation runner.
- **`system_builder.py`**: central “composition root” that builds and connects the components:
  - plant, controller, vision, estimator
  - optional mechanism, actuator, visualizer

This split keeps components decoupled and makes it easy to swap implementations (sim vs real, different estimators/controllers).

---

## Module responsibilities

### `core/` — dynamics, control, shared types

- **`core/sim_types.py`**: shared dataclasses that define contracts between modules:
  - `SystemState` (8D state)
  - `TableCommand`, `TableAccel`
  - `PhysicalParams` (composition of `PlantParams`, `WorkspaceParams`, `MechanismParams`, `HardwareParams`, `RunParams`)
  - `HardwareParams.dvs_algo`: `"sam"` or `"hough"` — selects DVS line algorithm
  - `BenchmarkVariant` (controller/estimator/noise/delay for benchmark sweeps)
  - `ExperimentSetup` (bundles params, camera_params, default_variant)
  - benchmark result/summary types
- **`core/plant.py`**: `BalancerPlant`
  - table dynamics + pencil dynamics
  - workspace constraint (`safe_radius`) and acceleration limiting (`max_acc`)
- **`core/model.py`**: `BuildLinearModel(params)` builds linear \(A,B\) used by controllers/observers.
- **`core/controller.py`**: controllers that map `SystemState` → `TableCommand`
  - `LQRController`, `PolePlacementController`, `NullController`

### `perception/` — vision, event processing, estimation

- **`perception/vision.py`**:
  - shared projection/reconstruction math (`VisionModelBase`)
  - simulated measurement model (`SimVisionModel`) with optional noise + delay buffering
  - simulated event camera interface (`SimEventCameraInterface`)
  - real event camera interface (`RealEventCameraInterface`) — opens two DAVIS346 via dv-processing, background threads read events and run the selected line algorithm, `get_observation(None)` returns latest line estimate
- **`perception/dvs_camera_reader.py`**: `DVSReader` — wraps dv-processing, opens DAVIS346 by serial, `get_event_batch()`, `close()`. Optional `BackgroundActivityNoiseFilter` when `noise_filter_duration_ms` is set (e.g. 30 ms).
- **`perception/dvs_algorithms.py`**: event-to-line estimation algorithms (selected via `HardwareParams.dvs_algo`):
  - **`SamLineAlgorithm`** (default for real DVS) — see below
  - **`PaperHoughLineAlgorithm`** — paper-inspired quadratic Hough tracker
  - **`SurfaceRegressionAlgorithm`** — activity surface + regression (not used in main pipeline)
- **`perception/estimator.py`**: state estimators (pose → 8D `SystemState`)
  - `FiniteDifferenceEstimator`, `LowPassFiniteDifferenceEstimator`, `KalmanEstimator`
- **`perception/camera_model.py`**: camera geometry + pixel/normalized conversions.

#### DVS line algorithms and camera-line pipeline (2026-03)

- **Line model**: A camera line is represented as \(x = s\,y + b\) in both normalized and pixel coordinates (slope \(s\), intercept \(b\)).
- **Camera conversion**:
  - `CameraModel.camnorm_to_pixel()` applies
    - \(s_{px} = \frac{f_x}{f_y}\, s\)
    - \(b_{px} = f_x\, b - \frac{f_x}{f_y}\, s\, c_y + c_x\)
  - `CameraModel.pixel_to_camnorm()` uses the **consistent inverse**
    - \(b = \frac{b_{px} - c_x + s_{px} c_y}{f_x}\)
    - \(s = s_{px}\,\frac{f_y}{f_x}\)

**Sam algorithm (`SamLineAlgorithm`)** — primary algorithm for real DVS:

- **What it is**: Ordinary least-squares (OLS) line fit on event coordinates. Originated from the standalone `sam_cam.py` script. Fits directly to each event batch; no surface accumulation or temporal decay.
- **How it works**: Given events \((x_i, y_i)\), fits \(x = \text{slope} \cdot y + \text{intercept}\) via closed-form OLS. Requires `min_points=50` events per batch. Returns `CameraObservation(slope, intercept)` in pixel space.
- **How we use it**: Selected when `HardwareParams.dvs_algo="sam"` (default in `main.py`). Used with real DAVIS346 cameras via `RealEventCameraInterface`. When Sam is selected, `DVSReader` uses `BackgroundActivityNoiseFilter` (30 ms) to reject transient noise before events reach the algorithm. The filtered event batch is passed to `SamLineAlgorithm.update()`; the resulting pixel-space line is converted to normalized coordinates and fed into the standard pipeline: reconstruction → estimator → controller.
- **Integration**: Same interface as other algorithms (`update(events_np)` → `CameraObservation`). Works with both `RealEventCameraInterface` (real hardware) and `SimEventCameraInterface` (synthetic events). Lower latency than surface-based methods because it fits per batch without accumulation.

**Hough algorithm (`PaperHoughLineAlgorithm`)** — paper-inspired tracker:

- Maintains quadratic accumulators over centered pixel coordinates \((x - c_x, y - c_y)\).
- For each event: solves the current line, computes the event's residual, applies a Gaussian inlier weight, performs adaptive forgetting proportional to that weight, and accumulates the weighted event into the quadratic state. This mirrors the original `polyAddEventX/Y` logic in `Conradt/PencilBalancer.java`.
- The per-event inner loop is compiled via Numba (`@numba.njit`) in `_hough_update_events_jit()`. This brings throughput from ~30k-100k events/s (pure Python) to millions of events/s, matching the compiled-Java performance of the original system.
- Accepts an optional `max_events` parameter. When set, only the newest N events from a batch are processed by the algorithm (tail slice), acting as a safety cap under extreme event bursts.
- Selected when `dvs_algo="hough"`. No noise filter. Used in `benchmarks/benchmark_hough.py` for static/falling pencil and decay sweeps, and in `benchmarks/visualize_dvs_cams.py` for live camera validation.

**Surface algorithm (`SurfaceRegressionAlgorithm`)** — not used:

- Activity surface with decay; thresholds and morphology; OLS on surface points. Available but not wired into the main pipeline. For low latency, per-batch OLS (Sam) is preferred over surface accumulation.

**Benchmark (`benchmarks/benchmark_hough.py`)**:
- Generates synthetic DVS-like events, runs `PaperHoughLineAlgorithm`, measures pixel and normalized-space errors. Typical mean normalized intercept error \(\mathcal{O}(10^{-3})\), slope error \(\mathcal{O}(10^{-4})\).

**Live camera benchmark (`benchmarks/visualize_dvs_cams.py`)**:
- Opens two DAVIS346 cameras, runs Sam or Hough, renders event surfaces with line overlay at 30 FPS.
- Uses batch draining: each iteration reads all queued batches per camera, concatenates them, updates the display surface with every event, and feeds the merged batch to the algorithm. This prevents backlog buildup under high event rates.
- Supports `--max-events-per-batch N` to cap events fed to the Hough algorithm per iteration.
- Requires `sudo` for USB camera access: `sudo .venv/bin/python -m hardware.visualize_dvs_cams --mode hough`.

### `simulation/` — loop integration and scheduling

- **`simulation/simulator.py`**: single step:
  - plant step (or skipped when `real_mode`) → measurement → reconstruction → estimation → controller update (at actuator rate)
  - When `real_mode` (real DVS connected): no plant, vision → estimator → controller; returns `state_est`
- **`simulation/simulation_runner.py`**: runs the loop for `total_time`:
  - `dvs_cams_connected(params)` → `real_mode=True` → Simulator skips plant
  - optional real-time pacing
  - optional actuator output
  - optional rendering
  - logs state + acceleration histories

### `hardware/` — real-world actuation & calibration

- **`hardware/Servo_System.py`**: actuator interface for sending commands to the servo mechanism.
- Additional Arduino calibration/experiment scripts and firmware (`ArduinoFile.cpp`).

### `fivebar/` — mechanism kinematics

If enabled via `PhysicalParams` (e.g., base points `O`, `B`, link lengths `la`, `lb`):

- **`fivebar/transform.py`**: `FiveBarTransform` for coordinate frame transforms.
- **`fivebar/mechanism.py`**: `FiveBarMechanism` inverse/forward kinematics and configuration selection.

### `visualization/` — rendering

- **`visualization/realtime_visualizer.py`**: OpenCV windows showing the tracked lines per camera in real time.
- **`visualization/visualizer3d.py`**: 3D visualization utilities.

### `analysis/`, `benchmarks/`, `results/`, `utils/`

- **`results/`**: benchmark JSON outputs.
- **`utils/io_utils.py`**: save/load benchmark artifacts.
- **`analysis/graphing.py`**: load latest benchmark JSON and generate plots (robustness curves, heatmaps, tradeoffs).
- **`analysis/workspace_plotter.py`**: plot stability/effort vs workspace radius.
- **`benchmarks/`**: focused tests for subcomponents
  - `benchmark_hough.py` — static / falling pencil / decay sweep (Hough only)
  - `visualize_dvs_cams.py` — real DAVIS346 init, events + line overlay (Sam or Hough via `--mode`; verify before HIL)

---

## Simulation vs. real hardware selection

The system swaps implementations while keeping the same high-level contracts:

- **Vision**
  - Simulated: `SimVisionModel` (direct projection) or `SimEventCameraInterface` (event-like batches)
  - Real events: `RealEventCameraInterface` — DAVIS346 via `DVSReader`, background line algorithm (Sam or Hough per `dvs_algo`), `get_observation(None)` returns latest
- **Execution**
  - Sim: plant step → vision → estimator → controller
  - Real DVS (HIL): vision → estimator → controller (plant skipped)
- **Actuation**
  - Sim-only: no actuator
  - Hardware: `ServoSystem` (optionally using `fivebar/` kinematics)

The selection logic is in `system_builder.py`; `simulation_runner` checks `dvs_cams_connected(params)` to set `real_mode`.

**Real-DVS pause**: When real DVS cams are connected and realtime is on, **Space** toggles "system on" vs "system off". When paused, the actuator receives a center command (`TableCommand(x_ref, y_ref)`) every tick so the table stays at center; the workspace window shows a "System paused — table at center" overlay. **q** quits. The sim step still runs while paused so measurement/state stay fresh for unpause.

---

## References

- Conradt et al., "A Pencil Balancing Robot using a Pair of AER Dynamic Vision Sensors," IEEE, 2009
- `sam_cam.py` — standalone DVS line-tracking script (Sam OLS pipeline)
- `docs/TASKS/DVS_FULL_SIM.md` — DVS task status, latency notes, Hough vs Sam comparison

