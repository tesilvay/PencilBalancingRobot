# Inverted Pendulum System Overview

This repo implements a pencil-balancing inverted pendulum on a 2-DOF table. At a high level, the system does the same thing every cycle:

1. Propagate the plant forward one timestep.
2. Read the pencil pose from a sensor pipeline.
3. Estimate the full state, including velocities.
4. Choose which controller/estimator should currently drive the system.
5. Command the table through either a mock actuator or the real servo mechanism.

The core runtime loop lives in [src/system/system.py](/home/tomas/Documents/VSC/Purdue/src/system/system.py).

## What problem the system is solving

The state being controlled is:

- table/pencil tip position in `x` and `y`
- table/pencil tip velocity in `x` and `y`
- pencil tilt angle about `x` and `y`
- pencil angular velocity about `x` and `y`

Those are stored in `State(px, vx, ax, wx, py, vy, ay, wy)` in [src/shared.py](/home/tomas/Documents/VSC/Purdue/src/shared.py).

The control input is a desired table position:

- `ControlInput(px_cmd, py_cmd)`

The measurement coming from vision is only:

- `px`, `py`
- `ax`, `ay`

So the estimators are responsible for reconstructing the unmeasured velocities.

## End-to-end architecture

The runtime stack is assembled from registries and presets. The system preset defines:

- one or more plants
- one or more controllers
- one or more estimators
- one sensor
- one actuator
- one supervisor

That wiring is defined in [src/system/system.py](/home/tomas/Documents/VSC/Purdue/src/system/system.py), [src/system/__init__.py](/home/tomas/Documents/VSC/Purdue/src/system/__init__.py), and [src/shared.py](/home/tomas/Documents/VSC/Purdue/src/shared.py).

The important idea is that this repo can run in a few modes:

- pure simulation
- simulated DVS vision
- real DVS vision
- real servo hardware
- supervised startup/blended handoff modes

## Plant and dynamics model

The underlying linear model is built in [src/system/plant/dynamics_model.py](/home/tomas/Documents/VSC/Purdue/src/system/plant/dynamics_model.py).

Per axis, the state is:

- position `p`
- velocity `v`
- pencil angle `a`
- angular velocity `w`

The table is modeled as a second-order system with parameters:

- gravity `g`
- center-of-mass length `com_length`
- table time constant `tau`
- damping ratio `zeta`

The full 2D plant is just two copies of the 1D model, one for `x` and one for `y`.

There are two main plant implementations:

- [src/system/plant/balancer.py](/home/tomas/Documents/VSC/Purdue/src/system/plant/balancer.py): the real balancing model. Table acceleration directly affects pencil angular acceleration.
- [src/system/plant/placing.py](/home/tomas/Documents/VSC/Purdue/src/system/plant/placing.py): a "human is still holding the pencil" model used during acquisition/startup experiments. The table moves, but the pencil pose evolves independently from a noisy hand/tremor model.

So generally:

- `BalancerPlant` is the actual inverted-pendulum balancing plant.
- `PlacingPlant` is a pre-balancing/acquisition plant.

## Sensors and measurement pipeline

The sensor abstraction is the vision pipeline in [src/system/sensor/interface/base.py](/home/tomas/Documents/VSC/Purdue/src/system/sensor/interface/base.py).

Conceptually the repo uses the same structure as the original DVS pencil-balancing idea:

1. Two cameras observe the pencil from orthogonal views.
2. Each camera estimates a 2D line.
3. The two lines are combined into a 3D pencil measurement.
4. That produces `Measurement(px, py, ax, ay)`.

### Analytic sensor path

[src/system/sensor/interface/sim_analytic.py](/home/tomas/Documents/VSC/Purdue/src/system/sensor/interface/sim_analytic.py) is the clean simulation path.

It:

- projects the true state into two camera lines
- optionally adds noise and delay
- reconstructs `(px, py, ax, ay)` using the closed-form camera geometry

This is the simplest path for understanding the math.

### Simulated DVS path

[src/system/sensor/interface/sim_dvs.py](/home/tomas/Documents/VSC/Purdue/src/system/sensor/interface/sim_dvs.py) simulates event cameras more realistically.

It:

- projects the true pencil into two image-space lines
- generates synthetic events along those lines
- adds dropout, pixel noise, and background noise
- runs a line-tracking algorithm per camera
- converts the two tracked lines into a measurement

### Real DVS path

[src/system/sensor/interface/real_dvs.py](/home/tomas/Documents/VSC/Purdue/src/system/sensor/interface/real_dvs.py) is the real hardware pipeline.

It:

- opens two DAVIS346 event cameras
- reads asynchronous event batches in background threads
- masks out irrelevant regions
- runs a line tracker for each camera
- reconstructs the pencil measurement
- reuses the last valid observation if one camera briefly drops out

The camera hardware reader is in [src/system/sensor/reader/dvs_camera_reader.py](/home/tomas/Documents/VSC/Purdue/src/system/sensor/reader/dvs_camera_reader.py).

## What sensor algorithm they use

The main line-tracking algorithm is a recursive Hough-based tracker in [src/system/sensor/algo/hough.py](/home/tomas/Documents/VSC/Purdue/src/system/sensor/algo/hough.py).

That algorithm:

- maintains a quadratic objective over line slope/intercept
- updates it per event
- weights events by distance to the current line estimate
- forgets old evidence gradually with a mixing factor

So the general sensor strategy is:

- use two DVS cameras
- estimate one line per camera with a recursive Hough tracker
- combine the two lines into 3D pencil position and tilt

There is also support for alternative algorithms like `sam`, but the main repo presets use the Hough path.

## Observation / reconstruction model

There are two ways this repo turns two camera lines into a measurement:

1. Geometric closed-form reconstruction in [src/system/sensor/interface/base.py](/home/tomas/Documents/VSC/Purdue/src/system/sensor/interface/base.py) and [src/system/sensor/interface/sim_analytic.py](/home/tomas/Documents/VSC/Purdue/src/system/sensor/interface/sim_analytic.py).
2. A calibrated regression model in [src/system/sensor/observation_model/simple_dvs.py](/home/tomas/Documents/VSC/Purdue/src/system/sensor/observation_model/simple_dvs.py) and [src/system/sensor/observation_model/simple_dvs_regression_model.py](/home/tomas/Documents/VSC/Purdue/src/system/sensor/observation_model/simple_dvs_regression_model.py).

The regression model is especially important for the DVS pipelines. It learns a mapping from per-camera line features in pixel space to:

- table position `X`, `Y`
- tilt `alpha_x`, `alpha_y`

It also clamps outputs to reasonable tilt and workspace bounds.

So, generally:

- in ideal simulation, the repo can use exact geometry
- in DVS-based operation, it often prefers a calibrated regression model on top of the tracked lines

## Estimators

The estimators live in [src/system/estimator/__init__.py](/home/tomas/Documents/VSC/Purdue/src/system/estimator/__init__.py).

All estimators receive:

- the measured pose `(px, py, ax, ay)`
- the last applied command
- the timestep

They output:

- a full 8-state estimate
- an innovation/residual vector

### 1. Finite-difference estimator

[src/system/estimator/fde.py](/home/tomas/Documents/VSC/Purdue/src/system/estimator/fde.py)

This is the simplest estimator:

- positions/angles come straight from measurement
- velocities are estimated by finite differencing consecutive measurements

### 2. Low-pass finite-difference estimator

[src/system/estimator/lpf.py](/home/tomas/Documents/VSC/Purdue/src/system/estimator/lpf.py)

This is a smoothed version of the finite-difference idea:

- low-pass filters the measured channels
- finite-differences the filtered signal
- low-pass filters the velocity estimate too

This appears to be the "fast / robust / startup-friendly" estimator.

### 3. Kalman estimator

[src/system/estimator/kalman.py](/home/tomas/Documents/VSC/Purdue/src/system/estimator/kalman.py)

This estimator uses:

- the discrete-time linear plant model
- a measurement matrix that observes `px`, `ax`, `py`, `ay`
- process noise `Q`
- measurement noise `R`

It predicts with the model and corrects with the measurement, so it is the most model-based estimator in the repo.

### General estimator strategy

The most important pattern in this codebase is not just "pick one estimator." In the supervised modes, the system keeps multiple estimators running at once and can blend between them.

Usually that means:

- estimator 0: low-pass / finite-difference style estimator
- estimator 1: Kalman estimator

The supervisor can gradually ramp a blend factor `est_k` from `0` to `1`, so the controller transitions from trusting the simpler estimator to trusting the Kalman estimator.

That blending logic is implemented in [src/system/system.py](/home/tomas/Documents/VSC/Purdue/src/system/system.py), [src/system/supervisor/dynamic.py](/home/tomas/Documents/VSC/Purdue/src/system/supervisor/dynamic.py), and [src/system/supervisor/real_dynamic.py](/home/tomas/Documents/VSC/Purdue/src/system/supervisor/real_dynamic.py).

## Controllers

The controllers live in [src/system/controller/__init__.py](/home/tomas/Documents/VSC/Purdue/src/system/controller/__init__.py).

### 1. Pole placement controller

[src/system/controller/pole.py](/home/tomas/Documents/VSC/Purdue/src/system/controller/pole.py)

This is a classic full-state feedback controller:

- build linear model `A, B`
- choose desired closed-loop poles
- compute gain `K`
- apply `u = u_ref - K(x - x_ref)`

### 2. Smooth pole placement controller

[src/system/controller/smooth_pole.py](/home/tomas/Documents/VSC/Purdue/src/system/controller/smooth_pole.py)

This is the main balancing controller in the current presets.

Instead of directly commanding `u`, it:

- augments the state with the previous command
- solves for `delta u`
- adds explicit slew-related poles

That makes command changes smoother and better behaved for the physical table/servo system.

This is probably the controller to treat as the repo's main practical balancing controller.

### 3. LQR controller

[src/system/controller/lqr.py](/home/tomas/Documents/VSC/Purdue/src/system/controller/lqr.py)

This is another full-state linear controller using quadratic cost weights instead of explicit pole placement.

### 4. Null and circle controllers

- [src/system/controller/null.py](/home/tomas/Documents/VSC/Purdue/src/system/controller/null.py): outputs zero command, useful during acquisition or supervisory startup.
- [src/system/controller/circle.py](/home/tomas/Documents/VSC/Purdue/src/system/controller/circle.py): generates a test circular motion.

### General controller strategy

In the main balancing modes, the controller approach is:

- use full-state feedback based on the linearized model
- keep the desired state at workspace center with zero tilt
- command table motion that moves under the pencil to keep it upright

In the newer presets, the preferred controller is the smooth pole-placement controller, not a simple PD controller.

## Supervisors and startup logic

The supervisors decide:

- which controller is active
- which estimator dominates
- whether measurements should be zeroed/latching offsets
- whether startup/manual-centering behavior should override the controller

The base behavior is in [src/system/supervisor/base.py](/home/tomas/Documents/VSC/Purdue/src/system/supervisor/base.py).

### Static supervisor

[src/system/supervisor/static.py](/home/tomas/Documents/VSC/Purdue/src/system/supervisor/static.py)

This just picks a fixed controller and estimator.

### Dynamic supervisor

[src/system/supervisor/dynamic.py](/home/tomas/Documents/VSC/Purdue/src/system/supervisor/dynamic.py)

This is a software-only state machine with states like:

- `ACQUISITION`
- `STABILIZATION_READY`
- `STABILIZING`
- `BALANCED`

It watches:

- whether the pencil is stable enough
- whether the innovation gets too large
- whether the two estimators agree closely enough

Then it:

- switches from acquisition controller to balancing controller
- gradually blends from estimator 0 to estimator 1

### Real supervisor

[src/system/supervisor/real.py](/home/tomas/Documents/VSC/Purdue/src/system/supervisor/real.py)

This adds operator-assisted startup for the real hardware:

- manual centering with keyboard nudges
- acquisition phase while the pencil is held upright
- release into balanced operation

### Real dynamic supervisor

[src/system/supervisor/real_dynamic.py](/home/tomas/Documents/VSC/Purdue/src/system/supervisor/real_dynamic.py)

This combines:

- real-world manual startup
- estimator agreement checks
- gradual LPF-to-Kalman blending

So generally, the supervisor layer is how the repo solves the hardest practical part of the real system: not just balancing, but safely getting into a state where balancing can begin.

## Actuation

The actuator layer turns table position commands into either simulation output or real mechanism motion.

- [src/system/actuator/mock.py](/home/tomas/Documents/VSC/Purdue/src/system/actuator/mock.py): simulation/no-op actuator
- [src/system/actuator/servo.py](/home/tomas/Documents/VSC/Purdue/src/system/actuator/servo.py): real serial-controlled servo actuator

The servo actuator sends commands through a five-bar mechanism model and serial link to the Arduino/servo hardware.

## What the main presets are doing

The main system presets are in [src/system/system.py](/home/tomas/Documents/VSC/Purdue/src/system/system.py).

### `simple_sim`

- plant: simulated balancing plant
- controller: smooth pole placement
- estimator: LPF
- sensor: analytic simulated vision
- actuator: mock
- supervisor: static

This is the simplest "closed-loop balancing in simulation" setup.

### `placing_only`

- plant: simulated balancing/placing experiment setup
- controller: smooth pole placement
- estimator: Kalman
- sensor: simulated DVS + Hough

This is useful for testing DVS-style sensing.

### `dynamic_sim`

- plants: placing plant, then balancing plant
- controllers: null, then smooth pole
- estimators: LPF, then Kalman
- supervisor: dynamic

This preset is a full acquisition-to-balancing handoff simulation.

### `real_supervised`

- real DVS vision
- real servo actuator
- controllers: null for centering/acquisition, then smooth pole for run mode
- estimators: LPF and Kalman
- supervisor: real startup supervisor

### `real_dynamic_supervised`

This is the most complete real-system preset:

- real DVS sensor
- real servo actuator
- startup supervision
- estimator blending from LPF to Kalman
- smooth pole-placement balancing controller

## One-cycle walkthrough

The best mental model for `System.step()` in [src/system/system.py](/home/tomas/Documents/VSC/Purdue/src/system/system.py) is:

1. Use the currently active plant to advance the true state.
2. Ask the sensor for `Measurement(px, py, ax, ay)`.
3. Remove any workspace offset if the supervisor is still in acquisition/startup handling.
4. Run all estimators every step.
5. Blend the estimator outputs if the supervisor requests a handoff.
6. Feed the blended state to the active controller.
7. Allow the supervisor to override that command during startup/manual control.
8. Clamp the command to the workspace boundary.
9. Send the command to the actuator.
10. Let the supervisor update its internal state for the next cycle.

That loop is the main "why" of the file organization: almost every module exists to fill one slot in that cycle.

## File map: what each area is for

- [src/system/system.py](/home/tomas/Documents/VSC/Purdue/src/system/system.py): top-level closed-loop orchestration
- [src/shared.py](/home/tomas/Documents/VSC/Purdue/src/shared.py): common datatypes, presets, builders, workspace/reference utilities
- [src/system/plant/dynamics_model.py](/home/tomas/Documents/VSC/Purdue/src/system/plant/dynamics_model.py): linearized inverted-pendulum model used by controllers and Kalman
- [src/system/plant/balancer.py](/home/tomas/Documents/VSC/Purdue/src/system/plant/balancer.py): simulated balancing plant
- [src/system/plant/placing.py](/home/tomas/Documents/VSC/Purdue/src/system/plant/placing.py): acquisition / hand-held plant
- [src/system/sensor/interface/sim_analytic.py](/home/tomas/Documents/VSC/Purdue/src/system/sensor/interface/sim_analytic.py): idealized simulated vision
- [src/system/sensor/interface/sim_dvs.py](/home/tomas/Documents/VSC/Purdue/src/system/sensor/interface/sim_dvs.py): simulated DVS event-camera pipeline
- [src/system/sensor/interface/real_dvs.py](/home/tomas/Documents/VSC/Purdue/src/system/sensor/interface/real_dvs.py): real event-camera pipeline
- [src/system/sensor/algo/hough.py](/home/tomas/Documents/VSC/Purdue/src/system/sensor/algo/hough.py): recursive Hough line tracker
- [src/system/sensor/observation_model/simple_dvs_regression_model.py](/home/tomas/Documents/VSC/Purdue/src/system/sensor/observation_model/simple_dvs_regression_model.py): learned/calibrated line-to-state reconstruction
- [src/system/estimator/lpf.py](/home/tomas/Documents/VSC/Purdue/src/system/estimator/lpf.py): fast smoothed finite-difference estimator
- [src/system/estimator/kalman.py](/home/tomas/Documents/VSC/Purdue/src/system/estimator/kalman.py): model-based Kalman estimator
- [src/system/controller/smooth_pole.py](/home/tomas/Documents/VSC/Purdue/src/system/controller/smooth_pole.py): main practical balancing controller
- [src/system/supervisor/dynamic.py](/home/tomas/Documents/VSC/Purdue/src/system/supervisor/dynamic.py): simulated handoff/blending supervisor
- [src/system/supervisor/real_dynamic.py](/home/tomas/Documents/VSC/Purdue/src/system/supervisor/real_dynamic.py): real startup + blending supervisor
- [src/system/actuator/servo.py](/home/tomas/Documents/VSC/Purdue/src/system/actuator/servo.py): real hardware command output

## Short answer

If you want the shortest possible summary of how this repo solves the inverted-pendulum problem:

- sense pencil position and tilt from two DVS cameras
- track one line per camera with a recursive Hough algorithm
- reconstruct `(px, py, ax, ay)`
- estimate full state with LPF and/or Kalman filtering
- stabilize with full-state feedback, mainly smooth pole placement
- use supervisor state machines to handle startup, acquisition, estimator handoff, and real-world safety
