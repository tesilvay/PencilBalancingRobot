# InvertedPendulum System Overview

This document summarizes how the separate `InvertedPendulum/` project works at a system level. It is meant to answer:

- what problem the code is solving,
- what controller structure it uses,
- what sensors and derived signals it depends on,
- how motion commands are generated,
- and which files own each part of the logic.

## What the system is doing

This project controls a 2D inverted pendulum mounted on a planar mechanism driven by two motorized arms. The overall job is:

1. move the two actuated joints to a known home pose,
2. detect which pendulum/beam is installed,
3. measure the pendulum tilt,
4. stabilize the pendulum by moving the platform underneath it,
5. optionally run a motion sequence while keeping the pendulum balanced.

Conceptually, the controller does not directly "push on the pendulum." Instead, it commands planar platform motion `(x, y)`, then converts that desired platform motion into the two motor joint angles `phi0` and `phi1` through inverse kinematics.

## Main architecture

The top-level execution lives in `InvertedPendulum/InvertedPendulum/InvertedPendulum.cs`.

It runs as a high-rate isochronous task and does three core things every cycle:

1. read/update measurements,
2. run either the balancing controller or a direct dynamic move mode,
3. advance a state machine for startup, calibration, balancing, motion sequences, and recovery.

The main state flow is roughly:

- `Startup`: clear faults and prepare the drive
- `Enabling`: enable both axes
- `MoveHomePos`: move motors to a known home configuration
- `BeamDetection`: identify the installed beam and wait until tilt is small
- `RegulatingDelay`: let filters settle
- `Calibration`: estimate zero offsets for tilt sensing
- `MoveSequence`: balance while running a scripted trajectory
- `Uncouple`: stop/recover if the pendulum is lost or motion limits are violated
- `MoveDynamic`: alternate demo mode that sends trajectory commands directly

## Sensors and measured quantities

The sensing logic is centered in `InvertedPendulum/InvertedPendulum/Beam.cs` and `InvertedPendulum/InvertedPendulum/Robot.cs`.

### 1. Motor encoders

The actual motor/joint angles come from encoder signals on axes 0 and 1. These are used in `Robot.ForwardKinematics(...)` to compute:

- platform position `xPos`
- platform position `yPos`
- platform orientation `xi`

So the platform center position is not measured by a separate Cartesian sensor. It is derived from motor encoder positions through geometry.

### 2. Pendulum tilt sensor

Pendulum inclination is computed from two analog-like channels:

- `Encoders_1.PhaseA`
- `Encoders_1.PhaseB`

in `Beam.CalcInclination()`.

These appear to be hall-sensor-based inclination channels. The code applies:

- a stored offset,
- a beam-dependent gain,
- and a coordinate rotation using the current platform orientation `xi`

to produce pendulum tilt in world-aligned axes:

- `xIncPos`
- `yIncPos`

### 3. Beam identification input

`Beam.IdentifyBeam()` uses `AnalogIn[0]` to determine which pendulum variant is installed:

- low/short beam,
- high/long beam,
- or very short beam mode.

That choice changes the model parameters, hall gains, offsets, and controller tuning.

## Estimation and derived states

This project does not use a Kalman filter, Luenberger observer, or other explicit full-state estimator.

Instead, it builds the needed state from measured and derived signals:

- platform position comes from forward kinematics on motor encoders,
- pendulum angle comes from hall sensor channels,
- pendulum angular velocity is estimated by discrete differentiation of tilt with a low-pass filter,
- controller states are assembled in a transformed "controller normal form."

### Tilt-rate estimation

`Beam.CalcInclination()` estimates tilt velocity as:

- filtered discrete derivative of `xIncPos`
- filtered discrete derivative of `yIncPos`

using `alphaVel` as the filter coefficient.

So the velocity estimate is simple and practical: differentiate the tilt signal, then smooth it.

### Controller state construction

`InvertedPendulum/InvertedPendulum/Controller.cs` constructs 5-state vectors for x and y:

- state 0: integral of position error
- state 1: platform position-related state
- state 2: platform velocity-related state
- state 3: pendulum-acceleration-related term from tilt angle
- state 4: pendulum-jerk-related term from tilt rate

More specifically, the code uses:

- `-g * tilt_angle`
- `-g * tilt_rate`
- platform position/velocity corrected by `l0 * tilt`

This is a model-based state transformation rather than an observer estimating hidden states.

## Controller structure

The balancing controller is implemented in `InvertedPendulum/InvertedPendulum/Controller.cs`, with gain generation in `InvertedPendulum/InvertedPendulum/Beam.cs`.

### Controller type

At a high level, this is a model-based state feedback controller with:

- integral action on position error,
- feedback on position, velocity, and higher-order terms,
- feedforward from the path planner's desired position, velocity, acceleration, and jerk.

The controller runs independently on x and y with the same structure.

### Control law

For each axis, the controller computes desired platform acceleration as:

- integral term
- plus proportional/state feedback on position error
- plus velocity error term
- plus acceleration error term
- plus jerk error term

The commanded acceleration is then saturated and numerically integrated to get:

- commanded platform velocity
- commanded platform position

Those Cartesian platform commands are then turned into motor-angle commands through inverse kinematics.

### Gain scheduling / beam-dependent tuning

`Beam.SetParameters(...)` recalculates controller gains whenever the beam type changes.

The tuning depends on:

- beam natural frequency `WN`
- equivalent pendulum length `l0`
- damping-like tuning parameter `D0`
- state controller bandwidth
- integrator bandwidth

This means the controller is not one fixed set of gains for every pendulum. It retunes itself based on which beam is installed.

### Safety and lock logic

The controller only starts once the beam is near upright (`LockInThreshold`), and it exits if:

- tilt exceeds `LockOutThreshold`,
- motor/joint software limits are violated,
- or inverse kinematics says the requested point is out of range.

## Motion planning

The motion-generation layer lives in:

- `InvertedPendulum/InvertedPendulum/PendulumPathPlanner.cs`
- `InvertedPendulum/InvertedPendulum/PathPlanner3rdOrder.cs`

### Reference generation

The balancing controller does not just hold a fixed point. It can track planned Cartesian references in x and y.

`PendulumPathPlanner` generates these reference trajectories:

- point-to-point moves,
- circle moves,
- Lissajous moves,
- hold/stay states.

For each axis, it produces reference:

- position
- velocity
- acceleration
- jerk

stored in `xState` and `yState`.

### Low-level trajectory primitive

`PathPlanner3rdOrder` is the core jerk-limited planner. It builds third-order motion profiles so the commanded motion is smooth and bounded by:

- maximum velocity,
- maximum acceleration,
- maximum jerk.

That smooth reference is important because the balancing controller explicitly uses up to jerk in its law.

## Kinematics and actuation

`InvertedPendulum/InvertedPendulum/Robot.cs` handles geometry.

### Forward kinematics

Given actual joint angles `phi0` and `phi1`, it computes:

- current platform center `(xPos, yPos)`
- platform orientation `xi`

This is used for sensing and calibration.

### Inverse kinematics

Given desired platform motion `(x, y)`, it computes joint commands:

- `phi0`
- `phi1`

The balancing controller therefore works in Cartesian space, while the motors are commanded in joint space.

### Output to the axes

`Controller.DoBackwardKinematics()` sends the balancing controller's Cartesian output to the motor path-planner stream registers.

`Controller.DoDynamicMoveStep()` instead converts the planned motion directly to joint commands, bypassing the balancing feedback law.

## Calibration

Calibration is handled in `Beam.CalcCalibValues()` and `Beam.CalibOffset()`.

The basic idea is:

- balance first,
- average the hall sensor channels for a while,
- treat that average as the zero-tilt reference,
- store the resulting offsets for the detected beam type.

This lets the same code work even if the installed pendulum or sensor alignment changes slightly.

## File-by-file responsibility map

- `InvertedPendulum/InvertedPendulum/InvertedPendulum.cs`: top-level real-time loop and state machine
- `InvertedPendulum/InvertedPendulum/Controller.cs`: balancing controller and Cartesian-to-joint command flow
- `InvertedPendulum/InvertedPendulum/Beam.cs`: beam detection, hall-sensor processing, tilt estimation, gain calculation, calibration
- `InvertedPendulum/InvertedPendulum/Robot.cs`: forward and inverse kinematics
- `InvertedPendulum/InvertedPendulum/PendulumPathPlanner.cs`: high-level reference trajectories and scripted demo sequence
- `InvertedPendulum/InvertedPendulum/PathPlanner3rdOrder.cs`: jerk-limited trajectory primitive
- `InvertedPendulum/InvertedPendulum/AxisHandler.cs`: axis enable/home/couple/uncouple/move helpers
- `InvertedPendulum/InvertedPendulum/Parameter.cs`: fixed physical constants, thresholds, limits, and nominal tuning values
- `InvertedPendulum/InvertedPendulum/TamaRegisters.cs`: named accessors for runtime parameters and debug/monitor variables
- `InvertedPendulum/InvertedPendulum/Utilities.cs`: warnings, limit checks, and shared `MoveState`

## Bottom line

The system solves the inverted-pendulum problem with a practical layered design:

- measure motor angles and pendulum tilt,
- derive platform position and tilt rate,
- use a beam-dependent model-based state feedback controller with integral action,
- generate smooth Cartesian references with a jerk-limited planner,
- convert Cartesian commands back into two motor joint commands through inverse kinematics,
- and supervise the whole process with a startup/calibration/recovery state machine.

If you read the code later, the most useful mental model is:

- `InvertedPendulum.cs` decides **when** the system should do something,
- `PendulumPathPlanner.cs` decides **where** the platform should go,
- `Controller.cs` decides **how to keep the pendulum stable while going there**,
- and `Robot.cs` converts between **platform motion** and **motor motion**.
