# Pencil Balancing Robot with Event Cameras (Python)

This project reimplements the pencil balancing robot described in

**J. Conradt et al., “A Pencil Balancing Robot using a Pair of AER Dynamic Vision Sensors,” IEEE, 2009** 

The original system used two Dynamic Vision Sensors (DVS) and a real-time Java implementation (jAER).
This repository reproduces the full pipeline in **Python**, including:

* Event-driven line tracking
* 3D pencil reconstruction from two orthogonal views
* Real-time control of a 2-DOF balancing table

---

## Why This Project Exists

Balancing a pencil on its tip is an extreme control problem.

For a 20 cm pencil, the angle increases ~10% every 10 ms under gravity .
That means:

* Millisecond-scale latency is mandatory.
* 60 Hz frame cameras are too slow.
* Standard vision pipelines introduce too much delay.

The original paper demonstrated that **event-based vision sensors** (DVS) solve this latency problem by emitting asynchronous brightness-change events instead of frames.

This project reconstructs that system architecture in Python.

---

## System Overview

The full system consists of four layers:

### 1. Event-Based Vision (DVS)

Each DVS emits timestamped events whenever pixel brightness changes.

Instead of processing frames:

* Each event updates the estimate immediately.
* Typical update intervals are in the hundreds of microseconds .

Two DVS cameras observe the pencil from orthogonal directions.

---

### 2. Line Tracking (Per Camera)

Each camera estimates a 2D line:

x = m·y + b

The original paper maintains a Gaussian in Hough space and updates it per event .

In this implementation:

* Each camera produces:

  * slope (s)
  * offset/intercept (b)

These two parameters represent the projected pencil.

---

### 3. 3D Pencil Reconstruction

Using the two line estimates (b₁, s₁) and (b₂, s₂), the system reconstructs:

* Pencil base position: (X, Y)
* Pencil tilt: (αₓ, αᵧ)

Using the closed-form solution from the paper :

```
X  = (b1·yr + b1·b2·xr) / (b1·b2 + 1)
αX = (s1 + b1·s2) / (b1·b2 + 1)
Y  = (b2·xr - b1·b2·yr) / (b1·b2 + 1)
αY = (s2 - b2·s1) / (b1·b2 + 1)
```

This yields a full 3D line estimate of the pencil.

---

### 4. Control

The original system used a PD controller .

This implementation currently supports:

* P control (position + tilt)
* Optional filtering (low-pass or Kalman)

Control law:

```
X_des = k_pos * X + k_tilt * αX
Y_des = k_pos * Y + k_tilt * αY
```

The goal is to keep:

* Position → 0
* Tilt → 0

Which corresponds to balancing upright at table center.

---

## Architecture

The Python code is structured for readability and modularity:

* `LineObservation` → 2D camera line
* `PencilState3D` → reconstructed state
* `PencilReconstructor` → triangulation logic
* `LowPassFilter` / `KalmanFilter1D` → filtering layer
* `PController2D` → control law
* `TableController` → filtered control output

All components are decoupled and testable.

---

## References

Conradt, J., Cook, M., Berner, R., Lichtsteiner, P., Douglas, R.J., Delbruck, T.
“A Pencil Balancing Robot using a Pair of AER Dynamic Vision Sensors,” IEEE, 2009. 
