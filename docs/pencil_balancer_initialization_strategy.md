# Pencil Balancer Initialization & Capture Strategy

## Problem Summary

You are not failing at control — you are failing at **initialization under invalid dynamics**.

At the moment of placing the pencil:
- The model is wrong
- The estimator is weak (LPF)
- The controller reacts poorly to noise
- The workspace is too large → system overreacts

Result: the table "runs away" instead of catching.

---

## Core Insight

Initialization is a **separate control problem**.

Treat it as a different mode:

1. **Capture phase (human-assisted, constrained system)**
2. **Stabilization phase (limited workspace, forgiving control)**
3. **Full control phase (normal LQR/Kalman)**

---

## Strategy 1: Progressive Workspace Expansion (Your Idea — Keep It)

### Concept

Start with a **very small allowed workspace**, then gradually expand it.

This limits controller authority early on, preventing runaway behavior.

### Why it works

- Early LPF estimates are noisy → large commands are dangerous
- Small workspace = bounded control output
- Human hand provides external stabilization initially

### Implementation

```python
workspace_radius = min_radius + growth_rate * t
workspace_radius = clamp(workspace_radius, min_radius, max_radius)

x_des = clamp(x_des, -workspace_radius, workspace_radius)
y_des = clamp(y_des, -workspace_radius, workspace_radius)
```

### Notes

- `min_radius` should be VERY small (almost fixed point)
- Growth should be slow (~0.5–2 seconds to full range)

---

## Strategy 2: Gain Scheduling (Critical)

### Concept

Reduce controller aggressiveness during capture.

### Early phase

- Low gains
- Especially reduce velocity and angle gains

### Later phase

- Gradually increase to full gains

### Implementation

```python
gain_scale = ramp(0.0 → 1.0 over 1–2 seconds)

K_effective = gain_scale * K_nominal
```

---

## Strategy 3: Estimator-Gated Control

### Concept

Don’t fully trust control until estimation is stable.

### Metrics to check

- Innovation small
- Velocity not exploding
- Angle within bounds

### Implementation

```python
if not estimator_stable:
    u = soft_control(lpf_state)
else:
    u = full_control(kalman_state)
```

---

## Strategy 4: Deadband Near Zero (Prevents Jitter)

### Problem

LPF noise causes table to move away from pencil.

### Fix

Ignore small errors.

```python
if abs(x_error) < epsilon:
    x_error = 0
```

### Effect

- Table stays under pencil instead of chasing noise

---

## Strategy 5: Velocity Limiting (You already discovered this)

### Keep it, but apply conditionally

- Strong limiting during capture
- Relax later

```python
u = clamp_rate(u, max_rate * gain_scale)
```

---

## Strategy 6: Human-Assisted Capture Mode

### Concept

Exploit the fact that your hand is stabilizing the pencil.

During this phase:
- The system does NOT need to fully balance
- It only needs to stay roughly underneath

### Simplified controller

```python
x_des = pose.X
```

No velocity, no angle compensation.

This avoids overreaction.

---

## Strategy 7: Explicit State Machine (Do This)

### Modes

```text
BOOTSTRAP → CAPTURE → STABILIZE → FULL_CONTROL
```

---

### BOOTSTRAP

- Estimator: LPF
- Workspace: tiny
- Gains: near zero

---

### CAPTURE

- Estimator: LPF
- Workspace: small
- Gains: low
- Human still holding

---

### STABILIZE

- Switch to Kalman (reset here)
- Workspace growing
- Gains increasing

---

### FULL_CONTROL

- Full workspace
- Full gains
- Kalman only

---

## Example State Machine

```python
if mode == "bootstrap":
    if measurement_stable():
        mode = "capture"

elif mode == "capture":
    if angle_small() and velocity_small():
        initialize_kalman()
        mode = "stabilize"

elif mode == "stabilize":
    if estimator_stable():
        mode = "full_control"
```

---

## Strategy 8: Artificial Damping Injection

### Problem

Early system is underdamped + noisy

### Fix

Add artificial damping term:

```python
u -= k_damp * x_dot
```

Even if velocity is noisy, small damping helps prevent runaway.

---

## Strategy 9: Delay Release (Human Timing Trick)

You are currently releasing too early.

Better approach:

1. Place pencil
2. Wait ~0.5s while system centers
3. Then release

You can even detect this automatically:

```python
if system_centered_for(300ms):
    allow_release = True
```

---

## Strategy 10: Clamp Angle During Capture

### Observation

Your hand is limiting angle physically.

Exploit that assumption:

```python
alpha_x = clamp(alpha_x, -alpha_limit, alpha_limit)
```

Prevents estimator spikes.

---

## Minimal Viable Setup (What I would actually implement)

If you want something that works quickly:

1. LPF estimator
2. Small workspace ramp
3. Gain ramp
4. Deadband
5. Kalman reset after ~0.5s of stable readings

Ignore everything else initially.

---

## Hard Truth

You’re trying to solve a **nonlinear capture problem with a linear controller**.

That will always be fragile unless you:

- constrain the system (workspace)
- reduce authority (gains)
- guide it through phases (state machine)

---

## Final Takeaway

Your idea is correct, but incomplete.

The real solution is:

> Controlled transition from a constrained, human-assisted system → autonomous system

Not just “wait until Kalman works.”

That alone won’t fix the capture problem.

