# Delta LQR Command Accumulation Plan

## Summary
Add a Delta LQR-only command accumulator so servo moves smaller than `2 mm` are not lost. Delta LQR will keep returning the last effective command until the signed accumulated 2D move has magnitude `>= 0.002 m`; then it will emit one absolute command containing the accumulated move and clear the pending accumulator.

## Key Changes
- Add a `command_deadzone: float = 2.0e-3` parameter to `DeltaLQRParams` and the `default` preset.
- Add controller state:
  - `_u_sent`: last command Delta LQR intentionally released to the actuator.
  - `_pending_delta_u`: signed accumulated controller deltas since `_u_sent`.
- Preserve the existing LQR math up through the raw candidate command:
  - Compute `delta_u` from LQR.
  - Apply existing `max_delta_u`.
  - Form candidate absolute command.
  - Apply existing command-radius limit.
- Apply accumulation after command-radius limiting:
  - Let `step_delta = candidate_u - _u_sent`.
  - Add `step_delta` into `_pending_delta_u`.
  - If `norm(_pending_delta_u) < command_deadzone`, return `_u_sent`.
  - If `norm(_pending_delta_u) >= command_deadzone`, return `_u_sent + _pending_delta_u`, clamp with `_limit_command_radius`, and clear pending after `set_applied_command()` confirms the command used.
- Alternating tiny commands cancel naturally because `_pending_delta_u` is signed:
  - `+1 mm`, then `-1 mm` accumulates back near zero and does not send.
  - Diagonal moves use 2D magnitude: `sqrt(dx^2 + dy^2) >= 2 mm`.

## Integration Details
- Keep accumulation inside `DeltaLQRController`, not `System` or `ServoActuator`, so the controller’s `_u_prev` tracks only commands that were actually released.
- Update `set_applied_command(u, state)`:
  - If the applied command differs from `_u_sent` by at least a tiny numerical tolerance, sync `_u_sent` and `_u_prev` to the applied command and clear `_pending_delta_u`.
  - If the command is the held command, keep `_u_sent` and `_u_prev` at that held value and leave `_pending_delta_u` intact.
  - Always continue updating the position and tilt reference integrators as it does now.
- Update `reset()` to clear `_pending_delta_u` and initialize `_u_sent`/`_u_prev` from the reset command/reference path.
- `reference_command()` remains the LQR reference command only; it should not include pending accumulated movement.

## Test Plan
- Add Delta LQR unit tests for:
  - A single `1 mm` delta returns the previous sent command and stores pending.
  - Two same-direction `1 mm` deltas emit a `2 mm` move and clear pending.
  - Alternating `+1 mm` then `-1 mm` returns the previous sent command and leaves no net pending move.
  - A diagonal accumulated move emits once its 2D norm reaches `2 mm`.
  - `reset()` clears pending accumulation and restores held command state.
  - Existing `max_delta_u` and command-radius limiting still apply before/around accumulation.
- Run:
  - `python3 -m py_compile src/system/controller/delta_lqr.py`
  - `pytest tests/test_delta_lqr.py`

## Assumptions
- The threshold is exactly `2.0e-3` meters.
- The threshold applies to signed accumulated 2D command magnitude, not per-axis thresholds.
- This feature belongs only to Delta LQR; other controllers and servo calibration/manual tools should keep sending commands exactly as they do now.
