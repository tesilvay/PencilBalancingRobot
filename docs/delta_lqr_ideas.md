# Bias Compensation Strategies for Pencil Balancer

## Problem

The system exhibits steady drift or off-center equilibrium even when the pencil is upright.

This typically comes from:
- actuator bias (table not neutral at `u = 0`)
- slight mechanical asymmetry
- small modeling errors

Current workaround (position reference integrator) is not ideal because:
- it shifts the *desired position*, not the physical cause
- it saturates
- it mixes multiple sources of bias

We want cleaner alternatives.

---

# 1. Command-Direction-Based Bias Learning (Improved Idea)

## Core Idea

Instead of using noisy velocity estimates or shifting position reference:

> Learn bias from **persistent command effort that maintains an offset**

---

## Intuition

If:
- the pencil is upright (tilt small)
- but remains off-center
- and the controller keeps applying a command in a consistent direction

Then:
> the system likely needs a constant offset (`u_bias`) to be neutral

---

## Key Signal

Use **held command**, not `delta_u`.

```python
u_hold = u_prev - u_ref_lqr
```

Seems to be we need trim input estimation
```python
u_bias += ki * pos_error * dt