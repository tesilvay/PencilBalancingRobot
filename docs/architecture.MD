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
  - real event camera interface (`RealEventCameraInterface`) — opens two DAVIS346 via dv-processing, background threads read events and run Hough, `get_observation(None)` returns latest line estimate
- **`perception/dvs_camera_reader.py`**: `DVSReader` — wraps dv-processing, opens DAVIS346 by serial, `get_event_batch()`, `close()`
- **`perception/dvs_algorithms.py`**: event-to-line estimation algorithms
  - `PaperHoughLineAlgorithm` (paper-inspired event-driven tracker)
  - `SurfaceRegressionAlgorithm` (activity surface + regression)
- **`perception/estimator.py`**: state estimators (pose → 8D `SystemState`)
  - `FiniteDifferenceEstimator`, `LowPassFiniteDifferenceEstimator`, `KalmanEstimator`
- **`perception/camera_model.py`**: camera geometry + pixel/normalized conversions.

#### Current Hough / camera-line pipeline (2026-03)

- **Line model**: A camera line is represented as \(x = s\,y + b\) in both normalized and pixel coordinates.
- **Camera conversion**:
  - `CameraModel.normalized_to_pixel()` applies
    - \(s_{px} = \frac{f_x}{f_y}\, s\)
    - \(b_{px} = f_x\, b - \frac{f_x}{f_y}\, s\, c_y + c_x\)
  - `CameraModel.pixel_to_normalized()` now uses the **consistent inverse**
    - \(b = \frac{b_{px} - c_x + s_{px} c_y}{f_x}\)
    - \(s = s_{px}\,\frac{f_y}{f_x}\)
  - This fixes a prior bug where the \(+s_{px} c_y\) term was missing, which biased the intercept while leaving the slope mostly correct.
- **Hough tracker (`PaperHoughLineAlgorithm`)**:
  - Maintains quadratic accumulators over centered pixel coordinates \((x - c_x, y - c_y)\).
  - On each event batch, decays the accumulators, integrates the new events, and solves a small quadratic form to recover slope \(m\) and a centered intercept, which is converted back to a pixel-space intercept \(b_{px}\).
- **Benchmark (`benchmarks/benchmark_hough.py`)**:
  - Generates synthetic DVS-like events by:
    - sampling \(y \sim U(0, H)\),
    - computing \(x = s_{px} y + b_{px}\),
    - adding pixel noise and **filtering to in-frame points**.
  - Runs many trials of `PaperHoughLineAlgorithm` and measures:
    - pixel-space line tracking quality,
    - normalized-space slope and intercept errors via the camera model.
  - With the fixed conversion and event generation, typical mean normalized intercept error is now \(\mathcal{O}(10^{-3})\) while slope error remains \(\mathcal{O}(10^{-4})\), which is accurate enough for the downstream pose reconstruction experiments.

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
  - `benchmark_hough.py` — static / falling pencil / decay sweep
  - `visualize_dvs_cams.py` — real DAVIS346 init, events + Hough line overlay (verify before HIL)

---

## Simulation vs. real hardware selection

The system swaps implementations while keeping the same high-level contracts:

- **Vision**
  - Simulated: `SimVisionModel` (direct projection) or `SimEventCameraInterface` (event-like batches)
  - Real events: `RealEventCameraInterface` — DAVIS346 via `DVSReader`, background Hough, `get_observation(None)` returns latest
- **Execution**
  - Sim: plant step → vision → estimator → controller
  - Real DVS (HIL): vision → estimator → controller (plant skipped)
- **Actuation**
  - Sim-only: no actuator
  - Hardware: `ServoSystem` (optionally using `fivebar/` kinematics)

The selection logic is in `system_builder.py`; `simulation_runner` checks `dvs_cams_connected(params)` to set `real_mode`.

