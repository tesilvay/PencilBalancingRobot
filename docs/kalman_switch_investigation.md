# Kalman Switch Investigation

## Problem

In `real_dynamic_supervised`, the first estimator (`lpf:test`) behaves acceptably, but when the supervisor switches to the second estimator (`kalman:default`), the estimated state appears to jump and the controller sends a large command to the table.

This does not happen in simulation to the same degree.

## Current Setup

- `real_dynamic_supervised` uses:
  - controllers: `[null:default, smooth_pole:smoother]`
  - estimators: `[lpf:test, kalman:default]`
  - supervisor: `real_dynamic:default`
- The real dynamic supervisor flow is:
  - `SERVO_CENTERING`
  - `ACQUISITION`
  - `STABILIZING`
  - `BALANCED`
- During `STABILIZING`, the run controller is already active.
- After `estimator_switch_delay_s`, the supervisor switches from estimator index `0` to estimator index `1`.

## Main Hypotheses

### 1. The Kalman filter is warm-started with a bad velocity estimate

At switch time, the system resets the new estimator using the current `x_hat`.

That means Kalman inherits:
- position and angle from LPF
- velocity and angular-rate estimates from LPF finite differences

The LPF velocity channels are derived from noisy measured signals, so they may be the least reliable part of the state. If those channels are wrong at the handoff, Kalman prediction may immediately drift and cause a large controller response.

### 2. The Kalman filter trusts the model too much on real hardware

The current Kalman preset may work in simulation because:
- the plant model is closer to truth
- measurements are cleaner
- timing is more ideal

On real hardware, the model may be less accurate and the vision measurement path is noisier, delayed, quantized, or occasionally stale. If the Kalman filter is too confident in prediction and not willing to move toward the real measurement quickly, then the first few post-switch estimates may be poor.

### 3. Reset covariance may be too small

When Kalman is reset at the switch, it starts from the passed-in state estimate but also resets covariance to a fixed initial matrix.

If that initial covariance is too small, Kalman behaves as if it is already quite confident in the inherited state. That makes it slower to correct any mismatch between LPF state and the actual real measured state.

### 4. Real sensor behavior may be harder for Kalman than LPF

The real DVS path can reuse prior valid measurements when a fresh observation is unavailable.

LPF can tolerate repeated or slightly stale measurements reasonably well because it is a simple smoothing estimator. Kalman assumes a cleaner state-space measurement/update process. On hardware, this mismatch may matter much more than in simulation.

### 5. The estimator switch is abrupt

The supervisor currently makes a hard estimator switch. There is no:
- state agreement check before switching
- temporary blend period
- guard that ensures Kalman has settled before becoming fully responsible

Even if Kalman is only slightly different from LPF, the controller can react strongly because the full state estimate changes in one step.

## Possible Fixes

## A. Tune the Kalman filter for real hardware

Possible tuning directions:

- Increase process noise on velocity states:
  - `q_vel_pos`
  - `q_vel_ang`
- Possibly increase process noise on measured state channels too:
  - `q_y_meas_pos`
  - `q_y_meas_ang`
- Revisit measurement noise:
  - `r_y_meas_pos`
  - `r_y_meas_ang`

Goal:
- make Kalman less stubborn about prediction
- help it adapt faster to real measurements after switching

This is the most direct fix if the issue is fundamentally estimator mismatch.

## B. Increase Kalman reset covariance at handoff

At switch time, Kalman could start with a larger `P` so it is less confident in the inherited state.

Goal:
- allow the first few real measurements to correct the estimate more aggressively
- reduce the effect of a slightly wrong LPF state at the handoff

This is especially useful if LPF position/angle are decent but velocity channels are questionable.

## C. Treat LPF velocities as less trustworthy at handoff

Instead of fully trusting all LPF state channels equally, the handoff could conceptually favor:
- LPF position and angle
- less confidence in LPF velocity and angular-rate terms

Ways this could be done later:
- inflate covariance for velocity-related states
- partially reset velocity states instead of copying them blindly
- use a different Kalman reset strategy when switching from LPF

This idea is motivated by the fact that LPF velocities come from finite differences on real vision signals.

## D. Switch only when LPF and Kalman agree

Before committing to the Kalman estimator, run a readiness check:
- compare Kalman and LPF estimates
- require their difference to remain below a threshold for some hold time

Possible agreement metrics:
- position difference norm
- angle difference norm
- maybe velocity difference norm

Goal:
- make sure Kalman is close enough before it becomes active
- avoid switching on a frame where Kalman would create a large discontinuity

This is a strong safety idea because it addresses estimator readiness directly.

## E. Blend the controller command during the transition

Professor suggestion:
- compute control from both filters
- smoothly blend `u_cmd` from LPF-based control to Kalman-based control over a short window

Possible benefit:
- reduces an abrupt command jump at the transition

Risk:
- it can hide a bad Kalman estimate instead of fixing it
- once blending ends, the same underlying estimator problem may still appear

This seems best as a second-stage smoothing mechanism, not the first fix.

## F. Combine agreement gating and blending

A stronger transition strategy could be:

1. Wait until LPF and Kalman agree sufficiently.
2. Switch responsibility.
3. Still blend `u_cmd` briefly to remove any remaining small step.

This is likely safer than blending alone.

## Recommended Order To Try

Recommended order for experiments:

1. Tune Kalman for real hardware:
   - especially increase process noise on velocity states
2. Increase reset covariance at handoff
3. Add a readiness / agreement check between LPF and Kalman before switching
4. If needed, add a short `u_cmd` blend window after switch

Reason:
- first fix the estimator
- then ensure it is ready before switching
- only then add smoothing if the remaining discontinuity is small

## Summary

The current best guess is that the problem is not just "Kalman is bad" in a general sense. It is more likely caused by some combination of:

- LPF-derived velocity state being poor at handoff
- Kalman trusting its inherited/model-based prediction too much
- real measurement behavior differing significantly from simulation
- an abrupt estimator switch with no readiness guard

The most promising next direction is to improve Kalman handoff robustness before relying on command blending.
