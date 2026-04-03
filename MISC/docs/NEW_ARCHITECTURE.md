# Architecture Guide

## Core Idea

Every class receives exactly one thing: a `Params` dataclass. That dataclass
holds everything the class needs — scalars, other dataclasses, or fully built
objects. The system that builds these params is `build_from_registry`. You
never call constructors manually.

The three pieces every class needs:

```
Presets dict      →  resolve_preset()  →  raw dict
raw dict          →  build_from_registry()  →  built objects / scalars
built objects     →  Params(**resolved)  →  Params dataclass
Params dataclass  →  cls(params)  →  final object
```

---

## The Building Blocks

### Spec

Every concrete class is registered in a `Spec`:

```python
@dataclass
class Spec:
    cls:        type            # the class to instantiate
    Params:     type            # the dataclass it receives
    Presets:    dict            # named preset dicts
    registries: dict | None     # field_name -> registry, for nested objects
    sim_only:   bool | None     # hardware compatibility flag
```

`registries` is the key field. It tells `build_from_registry` which registry
to use when it encounters a nested `"type:preset"` string in a preset dict.
If a field has no entry in `registries`, it is treated as a scalar and passed
through as-is.

### Presets

A preset is a flat dict of values for a class. Values are either:

- **Scalars** — `float`, `int`, `str`, `np.ndarray`, etc. Passed through directly.
- **`"type:preset"` strings** — signals a nested object. `build_from_registry`
  recurses into the matching registry to build it.
- **`{"name": "type:preset", ...}` dicts** — signals a dict of objects (e.g.
  multiple controllers). Each value is recursed independently.

Presets support inheritance via `"base"`:

```python
SERVO_PRESETS = {
    "default": {"port": "/dev/ttyUSB1", "baud": 115200, "frequency": 250.0},
    "fast":    {"base": "default", "frequency": 500.0},   # overrides one field
}
```

`resolve_preset` merges base into child recursively before building.

### build_from_registry

This is the only build function. It handles all nesting automatically:

```python
def build_from_registry(registry, spec_string):
    type_, preset = spec_string.split(":")
    spec = registry[type_]
    raw  = resolve_preset(spec.Presets, preset)

    resolved = {}
    for k, v in raw.items():
        sub_registry = (spec.registries or {}).get(k)

        if isinstance(v, str) and ":" in v:
            # nested object — recurse
            resolved[k] = build_from_registry(sub_registry, v)

        elif isinstance(v, dict) and sub_registry:
            # dict of objects — recurse each value
            resolved[k] = {
                name: build_from_registry(sub_registry, s)
                for name, s in v.items()
            }
        else:
            # scalar — pass through
            resolved[k] = v

    return spec.cls(spec.Params(**resolved))
```

The caller never needs to know how deep the nesting goes. Building an
experiment automatically builds its system, which builds its controllers,
which build their plant params — all from one call.

---

## Objects vs Shared Constants — Two Different Needs

`build_from_registry` always builds a full object from a `"type:preset"` string.
That covers live collaborators like `Mechanism` or `Estimator`. But some classes
only need physical constants — they never call methods on them. These are not
objects, they are shared data, and they should not go through a registry at all.

### Live collaborators — use the registry

A live collaborator is something the class calls methods on at runtime.
`Mechanism` is one: `servo.mechanism.command_to_angles(cmd)`. It belongs in
a registry, nested via `registries=`, built automatically.

```python
# servo calls methods on mechanism at runtime — it is a live collaborator
@dataclass
class ServoParams:
    mechanism: Mechanism   # built object
    port:      str
    frequency: float

ACTUATOR_REGISTRY = {
    "servo": Spec(ServoActuator, ServoParams, SERVO_PRESETS,
                  registries={"mechanism": MECHANISM_REGISTRY}),
}
```

### Shared constants — import directly from src/shared.py

`PlantParams` holds physical constants (`g`, `l`, `tau`, etc.) that controllers
and estimators need to compute matrices. They never call `.step()` on it — they
read numbers from it once in `__init__` and discard it. It is not a collaborator,
it is shared data.

These do not belong in a registry. They live in `src/shared.py` and are imported
directly. Controllers access them via a `default_factory` or by importing the
preset constants outright:

```python
# src/shared.py
@dataclass
class PlantParams:
    g: float
    l: float
    tau: float
    zeta: float
    num_states: int
    x_ref: float
    y_ref: float
    max_acc: float | None = None
    safe_radius: float | None = None

PLANT_PRESETS = {
    "default": {
        "g": 9.81, "l": 0.15, "tau": 0.03, "zeta": 0.8,
        "max_acc": 9.81 * 3, "num_states": 8,
        "x_ref": 0.0, "y_ref": 0.0, "safe_radius": 68e-3,
    }
}

def default_plant() -> PlantParams:
    return PlantParams(**PLANT_PRESETS["default"])
```

```python
# pole.py — controller uses PlantParams as pure data, no registry involved
from src.shared import PlantParams, default_plant
from dataclasses import dataclass, field

POLE_PRESETS = {
    "default": {"poles": [-14, -16, -18, -20] * 2},
    # no "plant" key — there is only one physical plant, no need to select it
}

@dataclass
class PoleParams:
    poles: list[float]
    plant: PlantParams = field(default_factory=default_plant)

class PolePlacementController:
    def __init__(self, params: PoleParams):
        A, B       = build_matrices(params.plant)   # reads numbers, done
        self.K     = place_poles(A, B, params.poles)
        self.x_ref = make_reference_state(params.plant.x_ref, params.plant.y_ref)

CONTROLLER_REGISTRY = {
    "pole": Spec(PolePlacementController, PoleParams, POLE_PRESETS),
    # no registries= needed — plant is not a nested object
}
```

The preset only contains what you actually tune (`poles`). `PlantParams` fills
in automatically from its default. If you ever need a different plant geometry,
you change `PLANT_PRESETS["default"]` in `src/shared.py` and every controller
and estimator sees it immediately.

### The decision rule

| Question | Answer |
|---|---|
| Do I call methods on it at runtime? | Registry + `registries=` |
| Do I just read constants from it in `__init__`? | Import from `src/shared.py`, `default_factory` |

Both look similar in a `Params` dataclass — one field is a live object, one is
a dataclass of scalars. The difference is entirely in how they are populated and
what the class does with them.

---

## Shared Params: src/shared.py

`PlantParams` and `TimingParams` are needed by multiple subsystems
(controllers, estimators, plant). If they lived inside `plant/`, importing
them would create circular imports. They live in `src/shared.py` instead.

`src/shared.py` also owns `Spec`, `resolve_preset`, and `build_from_registry`
since these are needed everywhere with no dependencies.

**Rule:** if a dataclass is imported by more than one subsystem, it lives in
`src/shared.py`.

---

## Nested Example: Servo → Mechanism

Servo needs a `Mechanism` object to do inverse kinematics. The user building
a system that uses a servo should not need to know anything about mechanisms.

```python
# mechanism.py
MECHANISM_PRESETS = {
    "default": {"O": np.array([0.0, 0.0]), "B": np.array([0.05, 0.0]),
                "la": 0.09, "lb": 0.09}
}

@dataclass
class MechanismParams:
    O: np.ndarray
    B: np.ndarray
    la: float
    lb: float

class Mechanism:
    def __init__(self, params: MechanismParams): ...
    def command_to_angles(self, command) -> tuple[float, float]: ...

MECHANISM_REGISTRY = {
    "five_bar": Spec(Mechanism, MechanismParams, MECHANISM_PRESETS),
}
```

```python
# servo.py
SERVO_PRESETS = {
    "default": {
        "mechanism": "five_bar:default",   # nested — built automatically
        "port":      "/dev/ttyUSB1",
        "baud":      115200,
        "frequency": 250.0,
    }
}

@dataclass
class ServoParams:
    mechanism:  Mechanism   # fully built object, not a string
    port:       str
    baud:       int
    frequency:  float

class ServoActuator(Actuator):
    def __init__(self, params: ServoParams):
        self.mechanism = params.mechanism   # already built, ready to use
        ...
```

```python
# actuator/__init__.py
ACTUATOR_REGISTRY = {
    "servo": Spec(ServoActuator, ServoParams, SERVO_PRESETS,
                  registries={"mechanism": MECHANISM_REGISTRY},  # wires the nesting
                  sim_only=False),
    "mock":  Spec(MockServoActuator, MockServoParams, MOCK_SERVO_PRESETS,
                  registries={"mechanism": MECHANISM_REGISTRY},
                  sim_only=True),
}
```

Building from the outside:
```python
actuator = build_from_registry(ACTUATOR_REGISTRY, "servo:default")
# mechanism was built automatically — caller never mentions it
```

The nesting is declared once in `registries=`. After that it is invisible.

---

## Nested Example: Vision → Interface → Algo + ObsModel

Vision is the most complex nesting case. It has three levels:

```
Vision
└── interface  (sim_analytic | sim_dvs | real_dvs)
    ├── algo       (hough | sam)
    └── obs_model  (none | simple_dvs)
```

### Level 3 — Algo and ObsModel (leaf classes, no nesting)

```python
# sensor/algo/__init__.py
LINE_ALGO_REGISTRY = {
    "hough": Spec(PaperHoughLineAlgorithm, HoughLineParams, HOUGH_PRESETS),
    "sam":   Spec(SamLineAlgorithm,        SamLineParams,   SAM_PRESETS),
}

# sensor/observation_model/__init__.py
REG_MODEL_REGISTRY = {
    "none":       Spec(NullRegression,           NullRegressionParams,   NULL_REG_PRESETS),
    "simple_dvs": Spec(SimpleDVSRegressionModel, SimpleRegressionParams, SIMPLE_REG_PRESETS),
}
```

### Level 2 — Interfaces (own algo + obs_model)

Each interface that processes raw events needs an algo and obs_model.
`sim_analytic` does not — it computes analytically and needs neither.

```python
# CORRECT
@dataclass
class RealDVSParams:
    algo:                     object        # built LineAlgorithm
    obs_model:                object        # built RegressionModel
    cam_params:               CameraParams  # scalar dataclass, no registry needed
    noise_filter_duration_ms: float | None = None
    cam1_device:              str   | None = None
    cam2_device:              str   | None = None

REAL_DVS_PRESETS = {
    "hough": {
        "algo":                     "hough:default",      # type:preset string → recurse
        "obs_model":                "simple_dvs:default", # type:preset string → recurse
        "cam_params":               "default:default",    # scalar dataclass → recurse into CAM_REGISTRY
        "noise_filter_duration_ms": None,
        "cam1_device":              None,
        "cam2_device":              None,
    },
    "sam": {"base": "hough", "algo": "sam:default", "noise_filter_duration_ms": 5},
}

# WRONG — this was the original bug:
# @dataclass
# class RealDVSParams:
#     algo:  object      ← fine
#     model: object      ← wrong name, preset uses "obs_model", causes TypeError on Params(**resolved)
#
# Preset key name must exactly match Params field name, case-sensitive.
# "obs_model" in preset + "model" in dataclass → TypeError: unexpected keyword argument 'obs_model'
```

```python
# sensor/interface/__init__.py
VISION_INTERFACE_REGISTRY = {
    "sim_analytic": Spec(
        SimVisionModel, SimAnalyticParams, SIM_ANALYTIC_PRESETS,
        sim_only=True,
        # no registries — sim_analytic has no nested objects
    ),
    "sim_dvs": Spec(
        SimEventCameraInterface, SimDVSParams, SIM_DVS_PRESETS,
        sim_only=True,
        registries={
            "algo":      LINE_ALGO_REGISTRY,
            "obs_model": REG_MODEL_REGISTRY,
        }
    ),
    "real_dvs": Spec(
        RealEventCameraInterface, RealDVSParams, REAL_DVS_PRESETS,
        sim_only=False,
        registries={
            "algo":      LINE_ALGO_REGISTRY,
            "obs_model": REG_MODEL_REGISTRY,
            "cam_params": CAM_REGISTRY,       # if CameraParams is also built via registry
        }
    ),
}
```

### Level 1 — Vision (owns interface)

`Vision` is a thin wrapper. It owns whichever interface was selected and
exposes a single `read()` method to the rest of the system. The interface
choice (and everything inside it) is already resolved before `Vision.__init__`
is called.

```python
VISION_PRESETS = {
    "sim_analytic":  {"interface": "sim_analytic:default"},
    "sim_dvs_hough": {"interface": "sim_dvs:hough"},
    "sim_dvs_sam":   {"interface": "sim_dvs:sam"},
    "real_dvs_hough":{"interface": "real_dvs:hough"},
    "real_dvs_sam":  {"interface": "real_dvs:sam"},
}

@dataclass
class VisionParams:
    interface: object    # fully built interface — any type, same API

class Vision:
    def __init__(self, params: VisionParams):
        self.interface = params.interface

    def read(self):
        return self.interface.read()

# sensor/__init__.py
VISION_REGISTRY = {
    "default": Spec(
        Vision, VisionParams, VISION_PRESETS,
        registries={"interface": VISION_INTERFACE_REGISTRY},
    )
}
```

Building vision from the outside — one call, zero knowledge of internals:
```python
vision = build_from_registry(VISION_REGISTRY, "default:real_dvs_hough")
# built: Vision → RealEventCameraInterface → PaperHoughLineAlgorithm
#                                          → SimpleDVSRegressionModel
#                                          → CameraParams
```

---

## The sim_only Flag

`sim_only` lives on the spec of whichever class touches real hardware — not
on composite wrappers. For actuator, it's on the actuator spec directly.
For vision, it's on the interface spec, because the algo and obs_model are
hardware-agnostic.

```python
def is_fully_sim(config):
    # actuator: one level, sim_only on the spec directly
    actuator_sim = ACTUATOR_REGISTRY[get_type(config["actuator"])].sim_only

    # vision: two levels deep — resolve vision preset to find which interface
    vision_raw     = resolve_preset(VISION_PRESETS, get_preset(config["vision"]))
    interface_type = get_type(vision_raw["interface"])
    vision_sim     = VISION_INTERFACE_REGISTRY[interface_type].sim_only

    return actuator_sim and vision_sim
```

---

## Rules Summary

| Rule | Reason |
|---|---|
| Preset key names must exactly match Params field names, case-sensitive | `Params(**resolved)` uses keyword unpacking — any mismatch is a `TypeError` |
| Scalars stay in presets, derived values go in `__init__` | Presets hold what you tune, classes hold what you compute |
| Live collaborators (call methods at runtime) go through registry + `registries=` | `build_from_registry` builds them automatically as nested objects |
| Shared constants (`PlantParams`, `TimingParams`) live in `src/shared.py`, not registries | They are data, not collaborators — import directly, use `default_factory` |
| `registries` declares nesting, `build_from_registry` executes it | Keeps build logic in one place, invisible to callers |
| `sim_only` lives on the hardware boundary spec, not on wrappers | Only the class that actually touches hardware decides sim compatibility |
| One registry per role, one entry per concrete class | Multiple entries only when behaviour differs enough to need a different class |
| `__init__.py` of a folder owns that folder's registry | Clean public interface — importers use the folder name, not internal file paths |
