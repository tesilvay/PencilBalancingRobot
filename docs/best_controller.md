# Controller-Only Parity Plan for `accel_pole`

## Summary
Bring `src/system/controller/accel_pole.py` closer to the reference controller by turning it from a raw acceleration integrator into a command-governed controller that only emits table trajectories the servos can plausibly follow.

This plan stays inside the controller/accel-architecture layer only. It does not add beam detection, calibration, external path planning, or supervisor state-machine work.

## Key Changes
### 1. Add a feasible-command governor inside `AccelPolePlacementController`
Replace the current “compute acceleration, clip once, integrate directly” flow with a bounded command state:
- Keep internal commanded states per axis: `cmd_pos`, `cmd_vel`, `cmd_acc`.
- Compute raw desired table acceleration from pole-placement feedback.
- Apply limits in this order each step:
  1. lock gating / safety hold
  2. jerk limit on change in commanded acceleration
  3. acceleration limit
  4. velocity limit
  5. workspace-aware position update
- Integrate with semi-implicit updates so the emitted `px_cmd/py_cmd` is the feasible result of the limited command state, not the unlimited feedback output.

### 2. Add missing safety features that prevent impossible maneuvers
Extend `AccelPoleParams` with:
- `max_vel_cmd: float | None`
- `max_jerk_cmd: float | None`
- `max_pos_error_for_int: float | None`
- `integrator_limit: float | None`
- `lock_in_angle_deg: float | None`
- `lock_out_angle_deg: float | None`
- `lock_out_hold_s: float`
- `hold_when_unlocked: bool`
- `workspace_margin: float`
- `boundary_brake_gain: float`

Required behavior:
- `lock_in`: controller integral and aggressive balancing stay disabled until tilt norm is below `lock_in_angle_deg`.
- `lock_out`: if tilt norm exceeds `lock_out_angle_deg` for `lock_out_hold_s`, freeze balancing action and hold the current feasible command or move toward workspace center depending on `hold_when_unlocked`.
- Integrator anti-windup:
  - do not integrate when output is saturated in a way that would increase saturation
  - clamp integral state to `integrator_limit`
  - optionally suppress integration when position-like error exceeds `max_pos_error_for_int`
- Workspace-aware braking:
  - before integrating position, compute radial distance to workspace center
  - reserve `workspace_margin` inside the workspace
  - when command velocity points outward near the boundary, apply inward braking proportional to `boundary_brake_gain`
  - still rely on external workspace clamp as a final guard, but make the controller avoid hitting it routinely

### 3. Make the controller state consistent with post-processing and resets
Update controller memory behavior:
- `set_applied_command()` must sync not only `cmd_pos` but also recompute `cmd_vel` and `cmd_acc` consistently from the applied trajectory history.
- `reset()` must clear lock state, saturation memory, fall/lock timers, and integral state.
- On warm start with `x_hat`, initialize `cmd_pos` from estimated table position and initialize `cmd_vel` from estimated table velocity; initialize `cmd_acc` to zero.

### 4. Improve the plant model only enough to validate controller behavior
Keep `src/system/plant/accel_follow.py` simple, but add the minimum support needed for testing the new controller assumptions:
- Preserve current `ideal` and `lagged` modes.
- In `ideal` mode, derive command velocity/acceleration from controller-emitted positions exactly as now.
- Add optional diagnostic outputs or test hooks so unit tests can inspect recovered command velocity/acceleration and verify the controller’s jerk/acc/velocity limiting is actually reflected at the plant input.

## Public API / Type Changes
Update `AccelPoleParams` in `src/system/controller/accel_pole.py` to include the new limiter and lock settings listed above.

No change to:
- `ControlInput`
- `State`
- external controller call signature

## Test Plan
Add controller-focused tests covering:
- raw pole-placement output requests excessive acceleration and the controller respects jerk then acceleration limits
- sustained acceleration would exceed max velocity and the controller caps velocity without runaway position integration
- integral state stops growing during saturation and remains within `integrator_limit`
- controller remains inactive before `lock_in` and activates once tilt is upright enough
- controller drops into lock-out when tilt exceeds threshold for the configured hold time
- near workspace boundary, outward commands are softened/braked before final clamp
- `set_applied_command()` keeps internal trajectory memory consistent after workspace clamping
- reset and warm-start produce bounded first-step commands with no large transient

## Assumptions
- “Impossible maneuvers” means excessive jerk, acceleration, velocity, and repeated slamming into workspace limits, not full mechanism-aware IK feasibility.
- Because this is controller-only scope, we will not add a separate jerk-limited path planner; the feasible-command governor inside `accel_pole` is the substitute.
- Workspace center remains the regulation target from `WorkspaceParams`.
- Any true mechanism/servo-angle feasibility beyond workspace radius remains an actuator/supervisor concern for a later phase.
