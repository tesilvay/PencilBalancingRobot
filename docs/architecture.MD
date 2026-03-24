# Project Architecture

This project models, simulates, and evaluates a 2-DOF table balancing an inverted pencil, with interchangeable perception pipelines (analytic and DVS) and experiment workflows (single run, Monte Carlo, benchmark sweeps, workspace sweeps, and real-time hardware runs).

The current architecture is organized around three core concepts:

- **Systems**: closed-loop runtime units (`plant + perception + controller`)
- **Builders**: composition functions that wire concrete implementations from config
- **Engines**: execution wrappers that initialize systems and run them through a common runner

---

## Runtime pipeline

At each simulation tick (`dt`), the loop is:

1. **Plant step** updates the true state from previous command.
2. **Perception** gets camera observation, reconstructs pose, and estimates full state.
3. **Controller** computes the next table command from estimated state.
4. **Runner services** actuator/render/logging/timing.

In code:

- `simulation/system.py`: `System.step(...)` performs plant → perception → control.
- `simulation/runner.py`: `ExperimentRunner.run()` handles stop logic, scheduling, pacing, optional actuator/rendering, and logging.

---

## Entry point and modes

The top-level entry point is `main.py`. It creates one `ExperimentSetup`, chooses an engine + experiment based on `--mode`, then executes.

Available modes:

- `single`: one finite simulation run + 3D playback
- `real`: real-time run with infinite stop policy (hardware-facing path)
- `montecarlo`: repeated randomized initial conditions, summarized metrics
- `benchmark`: grid of controller/estimator/noise variants over many trials
- `sweep`: workspace radius sweep, stability/effort tradeoff plot
- `graph`: post-processing/plotting of saved benchmark artifacts

Mode-to-engine defaults:

- `single`: `SimulationEngine(stop_policy=fixed_time)`
- `real`: `RealTimeEngine(stop_policy=infinite)`
- `montecarlo`, `benchmark`, `sweep`: `SimulationEngine(stop_policy=early_stop)`

---

## Core architecture layers

### 1) Configuration contracts (`core/sim_types.py`)

Dataclasses define all inter-module contracts:

- **State/IO**: `SystemState`, `TableCommand`, `TableAccel`
- **Vision**: `CameraObservation`, `CameraPair`, `PoseMeasurement`, `CameraParams`
- **Experiment config**: `PhysicalParams`, `BenchmarkVariant`, `ExperimentSetup`
- **Execution policy**: `StopPolicy`
- **Outputs**: `SimulationResult`, `TerminalInfo`, `TrialMetrics`, `BenchmarkSummary`, `BenchmarkResult`

`PhysicalParams` composes:

- `PlantParams` (dynamics constants and limits)
- `WorkspaceParams` (center and safe radius)
- `MechanismParams` (optional five-bar geometry)
- `HardwareParams` (servo + DVS mode and algorithm settings)
- `RunParams` (timing, tolerance, initial state randomization)

### 2) Dynamics and control (`core/`)

- `core/plant.py`: `BalancerPlant`
  - second-order table dynamics with damping
  - inverted-pendulum angular dynamics
  - command clamping to workspace boundary
  - acceleration magnitude limiting (`max_acc`)
  - boundary projection + outward velocity removal
- `core/model.py`: `BuildLinearModel(...)` returns linearized continuous `A, B` (8-state, 2-input)
- `core/controller.py`:
  - `LQRController`
  - `PolePlacementController`
  - `NullController`

### 3) Perception (`perception/`)

- `perception/vision.py`:
  - `VisionModelBase`: shared project/reconstruct geometry
  - `SimVisionModel`: analytic projection with optional Gaussian noise + delay buffer
  - `SimEventCameraInterface`: synthetic events -> line algorithms -> normalized camera observations
  - `RealEventCameraInterface`: dual-camera threaded ingestion with real DVS readers
  - `Perception`: vision + estimator wrapper producing `state_est`
- `perception/estimator.py`:
  - `FiniteDifferenceEstimator`
  - `LowPassFiniteDifferenceEstimator`
  - `KalmanEstimator` (discrete model from continuous `A, B`)
- `perception/dvs_algorithms.py`:
  - `PaperHoughLineAlgorithm` (recursive weighted quadratic tracker, Numba-accelerated update loop)
  - `SamLineAlgorithm` (batch OLS line fit in pixel space)
  - `SurfaceRegressionAlgorithm` (available but not the main path)

---

## Builders: composition root

The composition layer lives in `core/system_builder.py`.

### System build path

`system_factory(setup)` builds:

- plant (`build_plant`)
- controller (`build_controller`)
- estimator (`build_estimator`)
- vision backend (`build_vision`)
- perception wrapper (`Perception(vision, estimator)`)
- runtime system (`System(plant, perception, controller, dt)`)

### Vision build path

`build_vision(...)` dispatches by `params.hardware.vision_mode`:

- `sim_analytic` -> `SimVisionModel`
- `sim_dvs` -> `SimEventCameraInterface` with selected line algorithms
- `real_dvs` -> `RealEventCameraInterface` with discovered or specified camera devices

DVS algorithm selection is centralized in `build_line_algo(...)`:

- `"sam"`: two `SamLineAlgorithm` instances + optional noise filter duration
- `"hough"`: two `PaperHoughLineAlgorithm` instances with `HoughTrackerParams`

Optional learned pose regression is loaded for real DVS when enabled (`dvs_use_regression`).

### Runner build path

`runner_factory(params, system, stop_policy)` builds and injects:

- `Scheduler` (simulation tick + actuator/render rates)
- stop condition graph (`MaxSteps`, `FallCondition`, `StabilizedCondition`, `AnyStop`, `Infinite`)
- pacing (`NoPacing` or `RealTimePacing`)
- `Logger`
- optional five-bar mechanism + optional `ServoSystem` actuator
- optional realtime visualizer (`PencilVisualizerRealtime` or `DVSWorkspaceVisualizer`)

Result is one `ExperimentRunner`.

---

## Engines and execution

`simulation/engine.py` provides execution wrappers:

- `SimulationEngine.run(setup)`
- `RealTimeEngine` (currently same implementation, distinct type for mode semantics)

Engine responsibilities:

1. Build system and runner from setup.
2. Initialize randomized initial state around workspace reference (position + angle spreads from `RunParams`).
3. Initialize command to workspace center.
4. Delegate full loop to `ExperimentRunner.run()`.

`ExperimentRunner.run()` owns the main while-loop and produces `SimulationResult` + terminal status.

---

## Experiments and Monte Carlo workflows

Experiment classes are in `experiments/experiments.py`:

- `SingleExperiment`: one run, metric evaluation, optional 3D replay/video
- `RealExperiment`: one real-time run + summary
- `MonteCarloExperiment`: repeated runs with randomized initial conditions
- `BenchmarkExperiment`: variant sweep (`controller_type`, `estimator_type`, `noise_std`, `delay_steps`)
- `WorkspaceSweepExperiment`: workspace radius sweep, plots stability and average acceleration

Metric evaluation and summarization:

- `simulation/metrics.py`: per-trial `TrialMetrics`
- `experiments/utils.py`: aggregate into `BenchmarkSummary`

---

## Scheduling, stopping, pacing, logging

### Scheduler (`simulation/scheduler.py`)

- Tracks current simulated time `t`
- Independent cadence checks:
  - `should_actuate()` against `actuator_dt`
  - `should_render()` against `render_dt`

### Stop conditions (`simulation/stop_conditions.py`)

- `MaxSteps`: finite horizon
- `FallCondition`: angle-based failure
- `StabilizedCondition`: in-tolerance for a required settle window
- `AnyStop`: composed early-stop behavior
- `Infinite`: no stop (used for continuous real mode)

### Pacing (`simulation/pacing.py`)

- `NoPacing`: run as fast as possible
- `RealTimePacing`: wall-clock aligned stepping using `time.perf_counter()`

### Logging (`simulation/logger.py`)

- Stores state, command, acceleration histories
- Returns compact `SimulationResult` arrays at end of run

---

## Module map (quick reference)

- `main.py`: CLI mode selection and setup assembly
- `core/`: plant, linear model, controllers, dataclasses, builders
- `simulation/`: system loop, runner, engine, scheduler, stopping, pacing, metrics, logger
- `perception/`: camera geometry, DVS readers/algorithms, vision backends, estimators
- `experiments/`: high-level experiment orchestration and result summaries
- `hardware/`: servo interface and data/utility scripts for physical setup
- `fivebar/`: mechanism geometry and kinematics used by actuator/visualization paths
- `analysis/`: offline plotting/graph generation for benchmark outputs
- `visualization/`: realtime views and 3D replay tooling

---

## Current design intent

The architecture is intentionally layered so that:

- experiment logic is independent from low-level runtime loop mechanics,
- runtime loop mechanics are independent from specific perception backends,
- perception and control algorithms are swappable by config (`ExperimentSetup`),
- and simulation/hardware-adjacent workflows share one execution shape (`System` + `ExperimentRunner`).

This keeps Monte Carlo and benchmark workflows reproducible while still supporting real-time and hardware-oriented testing paths.

