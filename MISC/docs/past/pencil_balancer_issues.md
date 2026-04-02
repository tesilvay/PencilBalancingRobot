# Pencil Balancing Robot – Issues & Fixes

## 1. High-Frequency Instability on Placement

### Problem
- Pencil jitters and falls when placed.
- Caused by:
  - Initial velocity
  - High controller gains
  - Actuator delay
  - Sensor noise

### Fixes

#### Add actuator delay
```python
x_dot = (x_cmd - x) / tau  # tau ~ 20–30 ms
```

#### Add initial disturbance in sim
```python
theta = random_small_angle()
theta_dot = random_velocity()
```

#### Add measurement noise
```python
theta_measured = theta + noise()
```

#### Startup gain scheduling
```python
if t < 0.3:
    gains = low_gains
else:
    gains = normal_gains
```

---

## 2. Actuator Delay / Phase Lag

### Problem
- Commands act on outdated state → overshoot

### Fix

#### Model delay
```python
u_delayed = buffer.pop_oldest()
buffer.push(u)
```

---

## 3. Over-aggressive Control (Velocity Explosion)

### Problem
- Large commands → oscillations

### Fix

#### Low-pass control output
```python
u_filtered = alpha * u_prev + (1 - alpha) * u_raw
```

#### Increase damping
```python
u = gP*x + gA*theta + gD*x_dot  # increase gD
```

---

## 4. Friction Limit (Pencil Slipping)

### Problem
- Table acceleration exceeds friction → pencil flies off

### Physics
|a_table| ≤ μg

### Fix (Simulation)

#### Stick-slip model
```python
F_req = m * a_table
F_max = mu_s * m * g

if abs(F_req) <= F_max:
    x_base = x_table
else:
    F_friction = mu_k * m * g * sign(v_rel)
    x_base_ddot = F_friction / m
```

---

## 5. No Compliance (Rigid Contact Assumption)

### Problem
- Real system has deformation

### Fix

#### Spring-damper contact
```python
F = k*(x_table - x_base) + c*(v_table - v_base)

if abs(F) > mu_s * N:
    F = mu_k * N * sign(F)
```

---

## 6. No Acceleration Limits

### Problem
- Controller commands impossible motion

### Fix

#### Clamp acceleration
```python
a_cmd = clamp(a_cmd, -mu*g, mu*g)
```

---

## 7. No Capture Phase (Initial Contact)

### Problem
- Sim assumes immediate stable contact

### Fix

#### Gradual friction ramp
```python
mu = min(mu_s, mu_s * t / 0.1)
```

---

## 8. No Sampling / Discrete Effects

### Problem
- Real system is discrete with delays

### Fix

#### Discrete update
```python
x[k+1] = A*x[k] + B*u[k]
```

---

## 9. Missing Noise Spikes (Event Camera)

### Problem
- Sudden spikes cause overreaction

### Fix

#### Spike noise model
```python
if random_event():
    theta += spike_value
```

---

# Key Takeaways

- System is friction-limited, not ideal
- Control must respect physics constraints
- Simulation must include:
  - Delay
  - Noise
  - Friction
  - Saturation

Otherwise tuning is misleading
