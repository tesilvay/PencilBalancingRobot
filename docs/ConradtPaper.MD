# Pencil Balancing Robot Using Dynamic Vision Sensors

Based on: *A Pencil Balancing Robot using a Pair of AER Dynamic Vision Sensors* 

the code that they used is in the folder Conradt/

---

# 1. Overview

Balancing a pencil on its tip is a difficult control problem because the object becomes unstable extremely quickly. Small deviations grow exponentially, meaning the controller must react within **a few milliseconds**.

Traditional vision systems are too slow for this task because they rely on **frame-based cameras** (typically ~60 Hz). The delay between frames introduces latency that destabilizes the control loop.

This system solves the problem by using **Dynamic Vision Sensors (DVS)**, also called **event-based cameras**, which output events whenever brightness changes occur rather than sending full images.

Key properties of the system:

* Uses **two DVS cameras positioned orthogonally**
* Cameras detect motion of the pencil and output **asynchronous events**
* Events are processed immediately by a PC
* The PC estimates the **3D position and orientation of the pencil**
* A **PD controller** moves a table beneath the pencil to keep it upright

The system can balance small objects such as:

* pencils
* pens
* chopsticks
* rods

for **several minutes at a time**.

---

# 2. Why Balancing a Pencil is Difficult

Consider a pencil of length (L = 20) cm.

The center of mass of a uniform rod lies at:

[
c = \frac{L}{2}
]

For small angles ( \alpha ), the angular acceleration caused by gravity is:

[
\ddot{\alpha} = \frac{g}{c}\alpha
]

This means the tilt grows exponentially:

[
\alpha(t) \sim e^{t\sqrt{g/c}}
]

For a typical pencil:

* the tilt increases **10% every 10 ms**

Therefore the control system must react **within a few milliseconds** to stabilize it.

Frame-based cameras at 60 Hz produce updates every:

[
16.7 \text{ ms}
]

which is **too slow**.

Event-based cameras solve this problem by reacting **within microseconds**.

---

# 3. Dynamic Vision Sensor (DVS)

The system uses **two Dynamic Vision Sensors**.

Unlike normal cameras, a DVS does **not produce frames**.

Instead, each pixel independently reports a change in brightness.

Each event contains:

* pixel address
* polarity (increase or decrease in brightness)
* timestamp

Events are transmitted to the host computer as **Address Events (AEs)**.

Each event represents:

> A quantized change of log intensity at a pixel since the last event from that pixel. 

## Sensor Specifications

| Parameter               | Value                |
| ----------------------- | -------------------- |
| Dynamic range           | 120 dB               |
| Pixel array             | 128 × 128            |
| Pixel size              | 40 µm                |
| Photoreceptor bandwidth | ≥ 3 kHz              |
| Maximum event rate      | 1 million events/sec |
| Power consumption       | 23 mW                |

The camera sends events over **USB** in packets of up to **128 events**.

Typical packet interval:

```
~128 microseconds
```

---

# 4. Hardware System

The physical system is a **2-axis motion table** that moves the base of the pencil.

## Mechanical Structure

The table can move in **X and Y directions** across a **100 mm × 100 mm workspace**.

Mechanism:

1. Two parallel rollers allow motion along one axis
2. A third roller mounted perpendicular allows motion in the other axis
3. A small cart sits on the rollers
4. The cart holds a **cup with foam rubber**
5. The pencil rests inside the cup

The foam:

* prevents slipping during fast motion
* does not stabilize the pencil

Thus the pencil must be actively balanced.

---

## Actuation

Two **high-speed hobby servos** move the table:

| Axis   | Servo         |
| ------ | ------------- |
| X-axis | Futaba BLS451 |
| Y-axis | Futaba BLS251 |

Each servo connects through a **two-segment arm (2×100 mm)**.

These arms convert servo rotation into **linear motion**.

The arms are made from **machined aluminum** to reduce inertia.

---

## Position Sensing

Linear position sensors measure the cart position:

Device:

```
SpectraSymbol SoftPot SP-L-0100-101-ST
```

These sensors output a voltage proportional to position.

---

## Microcontroller

A **32-bit microcontroller (NXP LPC2103)** performs low-level control.

Properties:

* clock speed: 64 MHz
* reads position sensors at **8 kHz**
* applies **low-pass filtering**
* generates servo PWM signals

Servo PWM frequency:

```
250 Hz
```

Movement latency after a command:

```
~25 ms
```

The PC communicates with the microcontroller via **USB serial**.

Command rate:

```
~2 ms minimum interval
```

---

# 5. Software Architecture

Processing runs on a PC using the **jAER framework**, which is designed for event-based vision systems.

Two independent threads read data from the two cameras.

Each thread:

1. receives packets of events
2. processes them immediately
3. updates the estimate of the pencil position

Rendering and visualization run in a **lower priority thread**.

Typical CPU usage:

| Mode              | CPU load |
| ----------------- | -------- |
| No rendering      | 10–22%   |
| Rendering enabled | ~33%     |

The system processes roughly:

```
3 million events per second
```

while the actual event rate during balancing is:

```
200k – 300k events/sec
```

---

# 6. Pencil Tracking Algorithm

The tracking algorithm operates in **two stages**.

---

# Stage 1 — Line Estimation in Each Camera

Each camera sees the pencil as a **line in its image plane**.

The algorithm estimates the line parameters:

```
x = m y + b
```

Where:

* (m) = slope
* (b) = intercept

Instead of using classical Hough transform bins, the system maintains a **Gaussian distribution in Hough space**.

Hough space is defined by parameters:

```
(m, b)
```

The algorithm stores the Gaussian using **five quadratic coefficients**.

This compact representation avoids discretization.

---

## Event Update

Each incoming event at pixel ((x,y)) defines a line in Hough space.

The algorithm updates the quadratic coefficients using:

```
A = y²
B = 2y
C = 1
D = -2xy
E = -2x
```

Old data is gradually **decayed** and replaced with new event information.

---

## Extracting the Line

The estimated line corresponds to the **minimum of the quadratic**.

Slope and intercept are computed as:

[
(b,m) =
\left(
\frac{DB - 2AE}{q},
\frac{BE - 2CD}{q}
\right)
]

where

[
q = 4AC - B^2
]

---

# Stage 2 — 3D Reconstruction

Each camera provides:

* slope
* intercept

From these two observations the system reconstructs the **3D pencil line**.

Let the pencil line be:

[
(X + t\alpha_X, ; Y + t\alpha_Y, ; t)
]

From camera geometry the parameters can be solved as:

[
X = \frac{b_1 y_r + b_1 b_2 x_r}{b_1 b_2 + 1}
]

[
\alpha_X = \frac{s_1 + b_1 s_2}{b_1 b_2 + 1}
]

[
Y = \frac{b_2 x_r - b_1 b_2 y_r}{b_1 b_2 + 1}
]

[
\alpha_Y = \frac{s_2 - b_2 s_1}{b_1 b_2 + 1}
]

Where:

* (b_1, s_1) are intercept and slope from camera 1
* (b_2, s_2) from camera 2

The result is:

```
(X, Y)        → base position
(αX, αY)      → pencil tilt
```

---

# 7. Control System

The system uses a **PD controller**.

Inputs:

```
X, Y          pencil base position
αX, αY        pencil tilt
Ẋ, Ẏ        velocity
```

Outputs:

```
X_des
Y_des
```

These represent the **desired table position**.

Control law:

```
X_des = gP*X + gα*αX + gD*Xdot
Y_des = gP*Y + gα*αY + gD*Ydot
```

Where:

| Gain | Meaning       |
| ---- | ------------- |
| gP   | position gain |
| gα   | slope gain    |
| gD   | velocity gain |

Interpretation:

* **Position term** keeps the base centered
* **Slope term** anticipates falling direction
* **Velocity term** dampens motion

Typical gain values:

```
gP ≈ 1.3
gα ≈ 250
gD ≈ 70
```

---

# 8. System Performance

Typical update interval between event packets:

```
125 µs – 300 µs
```

Maximum observed delay:

```
9.8 ms
```

Median update interval:

```
236 µs
```

Thus the control loop runs **far faster than frame-based vision systems**.

---

## Motion Behavior

During balancing:

* the table stays near the center most of the time
* occasionally it uses the full range of motion

The measured delay between estimated pencil position and table response is about:

```
50 ms
```

This delay is the main factor limiting the maximum object length that can be balanced.

---

# 9. Lighting Conditions

The system works under a wide range of illumination:

* sunlight
* fluorescent light (~300 lux)
* incandescent light

Low light causes:

* increased sensor noise
* reduced pixel bandwidth

but the system remains stable if lighting is relatively steady.

---

# 10. Advantages of Event-Based Vision

A conventional camera solution would require:

```
>1000 FPS
```

to achieve similar latency.

At 128×128 resolution:

```
32 million pixels per second
```

must be processed.

Additional processing would be needed to:

* compute temporal contrast
* subtract previous frames
* detect motion

This would require **hundreds of millions of instructions per second**.

Event-based sensors avoid this by:

* producing data **only when changes occur**
* enabling **microsecond latency**
* reducing bandwidth dramatically

---

# 11. Key Insight

The project demonstrates that **event-driven sensing and computation** can dramatically simplify difficult real-time robotics tasks.

Instead of processing entire images continuously, the system processes only **meaningful changes**, allowing extremely fast feedback control similar to biological vision systems.