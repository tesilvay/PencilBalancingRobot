# Local UI Plan (Streamlit Wrapper for CLI)

## Objective
Create a local UI that replaces CLI flags for configuring and launching experiments.

The UI must act as a **thin wrapper** around the existing system.

---

## Core Principle

**Do NOT modify core logic.**

The UI should:
- Call existing functions (`make_preset`, `run_experiment`)
- Pass parameters
- Display basic outputs

The UI should NOT:
- Change experiment architecture
- Interfere with control loops
- Replace visualization systems

---

## Architecture

```
[Streamlit UI]
      ↓
(make_preset)
      ↓
(run_experiment)
      ↓
[OpenCV / matplotlib visualization]
```

UI is only a control layer.

---

## Requirements

### Inputs (UI Controls)

Minimum:
- Preset selection:
  - sim
  - hybrid
  - real

Optional (if easy to expose):
- controller_type
- estimator_type
- noise_std
- delay_steps

---

### Actions

- Button: **Run Experiment**

Behavior:
1. Build config using `make_preset` or equivalent
2. Call `run_experiment(setup)`
3. Show status/logs

---

### Outputs

- Text logs (stdout or returned values)
- Basic metrics (if available)

---

## Explicit Constraints (Important)

The UI MUST NOT:

- ❌ Replace OpenCV windows
- ❌ Replace matplotlib visualizations
- ❌ Stream frames to browser
- ❌ Modify real-time loops
- ❌ Introduce threading inside UI
- ❌ Change timing behavior

---

## Implementation Notes

- Use **Streamlit**
- Run locally via:

```
streamlit run app.py
```

- Keep UI code isolated:

```
ui/
  app.py
```

---

## Minimal Example

```python
import streamlit as st
from your_module import make_preset, run_experiment

preset = st.selectbox("Mode", ["sim", "hybrid", "real"])

if st.button("Run"):
    setup = make_preset(preset)
    results = run_experiment(setup)
    st.write(results)
```

---

## Future Improvements (Optional)

- Auto-generate UI from dataclasses
- Add parameter sliders/inputs
- Display structured metrics
- Add experiment history

---

## Summary

This UI is:
- A replacement for CLI flags
- A control panel for experiments

It is NOT:
- A visualization system
- A real-time rendering layer
- A redesign of the architecture

Keep it simple and non-invasive.

