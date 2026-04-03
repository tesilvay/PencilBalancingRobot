# Implementation Plan: New Architecture

## The Rule

Every folder follows the same pattern:

```
role/
    __init__.py     # registry lives here, re-exports public interface
    base.py         # ABC for this role (if it has multiple concrete types)
    type_a.py       # ConcreteClass + Params + PRESETS
    type_b.py       # ConcreteClass + Params + PRESETS
```

Composite classes (system, experiment) follow the same pattern — their
`__init__.py` just imports from child registries instead of sibling files.

Shared params used across subsystems (`PlantParams`, `TimingParams`) live in
`src/shared.py` so no circular imports occur.

---

## File Map

```
src/
    shared.py                       # PlantParams, PLANT_PRESETS, TimingParams, TIMING_PRESETS, Spec, build_from_registry, resolve_preset
    registries.py                   # imports and re-exports every registry (convenience hub)
    main.py

    system/
        __init__.py                 # SYSTEM_REGISTRY
        system.py                   # System, SystemParams, SYSTEM_PRESETS

        plant/
            __init__.py             # PLANT_REGISTRY  (re-exported from shared.py)
            balancer.py             # BalancerPlant, PlantParams, PLANT_PRESETS  → lives in shared.py, just imported here
            null.py                 # NullPlant, NullParams, NULL_PRESETS

        controller/
            __init__.py             # CONTROLLER_REGISTRY
            base.py                 # Controller ABC
            pole.py                 # PoleController, PoleParams, POLE_PRESETS
            lqr.py                  # LQRController, LQRParams, LQR_PRESETS
            smooth_pole.py          # SmoothPoleController, SmoothPoleParams, SMOOTH_POLE_PRESETS
            circle.py               # CircleController, CircleParams, CIRCLE_PRESETS
            null.py                 # NullController, NullParams, NULL_PRESETS

        estimator/
            __init__.py             # ESTIMATOR_REGISTRY
            base.py                 # Estimator ABC
            fde.py                  # FiniteDifferenceEstimator, FDEParams, FDE_PRESETS
            lpf.py                  # LowPassFiniteDifferenceEstimator, LPFParams, LPF_PRESETS
            kalman.py               # KalmanEstimator, KalmanParams, KALMAN_PRESETS
            full_kalman.py          # FullStateKalmanFilter, FullKalmanParams, FULL_KALMAN_PRESETS

        sensor/
            __init__.py             # SENSOR_REGISTRY  (wraps VISION_REGISTRY)
            vision.py               # Vision, VisionParams, VISION_PRESETS
            interface/
                __init__.py         # VISION_INTERFACE_REGISTRY
                base.py             # SensorInterface ABC
                sim_analytic.py     # SimVisionModel, SimAnalyticParams, SIM_ANALYTIC_PRESETS
                sim_dvs.py          # SimEventCameraInterface, SimDVSParams, SIM_DVS_PRESETS
                real_dvs.py         # RealEventCameraInterface, RealDVSParams, REAL_DVS_PRESETS
            algo/
                __init__.py         # LINE_ALGO_REGISTRY
                base.py             # LineAlgorithm ABC
                hough.py            # PaperHoughLineAlgorithm, HoughLineParams, HOUGH_PRESETS
                sam.py              # SamLineAlgorithm, SamLineParams, SAM_PRESETS
            observation_model/
                __init__.py         # REG_MODEL_REGISTRY
                base.py             # RegressionModel ABC
                null.py             # NullRegression, NullRegressionParams, NULL_REG_PRESETS
                simple_dvs.py       # SimpleDVSRegressionModel, SimpleRegressionParams, SIMPLE_REG_PRESETS

        actuator/
            __init__.py             # ACTUATOR_REGISTRY
            base.py                 # Actuator ABC
            servo.py                # ServoController, ServoParams, SERVO_PRESETS
            mock.py                 # MockServoController, MockServoParams, MOCK_SERVO_PRESETS

        supervisor/
            __init__.py             # SUPERVISOR_REGISTRY
            base.py                 # Supervisor ABC
            dynamic.py              # DynamicSupervisor, DynamicSupervisorParams, DYNAMIC_SUPERVISOR_PRESETS
            static.py               # StaticSupervisor, StaticSupervisorParams, STATIC_SUPERVISOR_PRESETS

    experiment/
        __init__.py                 # EXPERIMENT_REGISTRY
        experiment.py               # Experiment, ExperimentParams, EXPERIMENT_PRESETS

        logger/
            __init__.py             # LOGGER_REGISTRY
            base.py                 # Logger ABC
            logger.py               # Logger, LoggerParams, LOGGER_PRESETS

        stop_condition/
            __init__.py             # STOP_CONDITION_REGISTRY
            base.py                 # StopCondition ABC
            fall.py                 # FallCondition, FallConditionParams, FALL_CONDITION_PRESETS
            stabilized.py           # StabilizedCondition, StabilizedParams, STABILIZED_CONDITION_PRESETS
            max_steps.py            # MaxStepsCondition, MaxStepsConditionParams, MAX_STEPS_CONDITION_PRESETS
            any_condition.py        # AnyStopCondition, AnyStopConditionParams, ANY_STOP_CONDITION_PRESETS
            infinite.py             # InfiniteCondition, InfiniteConditionParams, INFINITE_CONDITION_PRESETS

        visualizer/
            __init__.py             # VISUALIZER_REGISTRY
            base.py                 # Visualizer ABC
            sim_dvs.py              # SimDvsVisualizer, SimDvsVisualizerParams, SIM_DVS_VISUALIZER_PRESETS
            real_dvs.py             # RealDvsVisualizer, RealDvsVisualizerParams, REAL_DVS_VISUALIZER_PRESETS
            one_dvs.py              # OneDvsVisualizer, OneDvsVisualizerParams, ONE_DVS_VISUALIZER_PRESETS
            sim_dvs_workspace.py    # SimDvsWorkspaceVisualizer, ..Params, ..PRESETS
            real_dvs_workspace.py   # RealDvsWorkspaceVisualizer, ..Params, ..PRESETS
            visualizer_3d.py        # Visualizer3D, Visualizer3DParams, VISUALIZER_3D_PRESETS

        progress/
            __init__.py             # PROGRESS_REGISTRY
            base.py                 # Progress ABC
            console.py              # ConsoleProgress, ProgressParams, PROGRESS_PRESETS

        pacing/
            __init__.py             # PACING_REGISTRY
            base.py                 # Pacing ABC
            realtime.py             # RealTimePacing, RealTimePacingParams, REALTIME_PACING_PRESETS
            null.py                 # NoPacing, NullParams, NULL_PRESETS

        scheduler/
            __init__.py             # SCHEDULER_REGISTRY
            base.py                 # Scheduler ABC
            scheduler.py            # Scheduler, SchedulerParams, SCHEDULER_PRESETS

        metrics/
            metrics.py              # Metrics — no registry, not swappable, called directly in main
```

---

## What each `__init__.py` looks like

### Leaf folder (estimator)
```python
# system/estimator/__init__.py
from .base       import Estimator
from .fde        import FiniteDifferenceEstimator,          FDEParams,        FDE_PRESETS
from .lpf        import LowPassFiniteDifferenceEstimator,   LPFParams,        LPF_PRESETS
from .kalman     import KalmanEstimator,                    KalmanParams,     KALMAN_PRESETS
from .full_kalman import FullStateKalmanFilter,             FullKalmanParams, FULL_KALMAN_PRESETS
from src.shared  import Spec, PLANT_REGISTRY

ESTIMATOR_REGISTRY = {
    "fde":          Spec(FiniteDifferenceEstimator,        FDEParams,        FDE_PRESETS,        registries={"plant": PLANT_REGISTRY}),
    "lpf":          Spec(LowPassFiniteDifferenceEstimator, LPFParams,        LPF_PRESETS,        registries={"plant": PLANT_REGISTRY}),
    "kalman":       Spec(KalmanEstimator,                  KalmanParams,     KALMAN_PRESETS,     registries={"plant": PLANT_REGISTRY}),
    "full_kalman":  Spec(FullStateKalmanFilter,            FullKalmanParams, FULL_KALMAN_PRESETS,registries={"plant": PLANT_REGISTRY}),
}
```

### Composite folder (system)
```python
# system/__init__.py
from .system      import System, SystemParams, SYSTEM_PRESETS
from .controller  import CONTROLLER_REGISTRY
from .estimator   import ESTIMATOR_REGISTRY
from .sensor      import SENSOR_REGISTRY
from .actuator    import ACTUATOR_REGISTRY
from .supervisor  import SUPERVISOR_REGISTRY
from src.shared   import Spec

SYSTEM_REGISTRY = {
    "default": Spec(
        cls        = System,
        Params     = SystemParams,
        Presets    = SYSTEM_PRESETS,
        registries = {
            "controllers": CONTROLLER_REGISTRY,
            "estimators":  ESTIMATOR_REGISTRY,
            "sensor":      SENSOR_REGISTRY,
            "actuator":    ACTUATOR_REGISTRY,
            "supervisor":  SUPERVISOR_REGISTRY,
        }
    )
}
```

### Top-level convenience hub
```python
# src/registries.py  — nothing lives here, only imports
from .system     import SYSTEM_REGISTRY
from .system.controller  import CONTROLLER_REGISTRY
from .system.estimator   import ESTIMATOR_REGISTRY
from .system.sensor      import SENSOR_REGISTRY
from .system.actuator    import ACTUATOR_REGISTRY
from .system.supervisor  import SUPERVISOR_REGISTRY
from .experiment import EXPERIMENT_REGISTRY
from .experiment.logger         import LOGGER_REGISTRY
from .experiment.stop_condition import STOP_CONDITION_REGISTRY
from .experiment.visualizer     import VISUALIZER_REGISTRY
from .experiment.progress       import PROGRESS_REGISTRY
from .experiment.pacing         import PACING_REGISTRY
from .experiment.scheduler      import SCHEDULER_REGISTRY
```

---

## Shared params problem

`PlantParams` is needed by controllers, estimators, and the plant itself.
If it lives in `plant/balancer.py`, everyone importing it creates a potential
circular import. Put shared dataclasses in `src/shared.py` instead:

```python
# src/shared.py
from dataclasses import dataclass
import numpy as np

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

@dataclass
class TimingParams:
    total_time: float = 5.0
    dt:         float = 4e-3

TIMING_PRESETS = {
    "default": {"total_time": 5.0, "dt": 4e-3},
    "long":    {"total_time": 30.0, "dt": 4e-3},
}

@dataclass
class Spec:
    cls:        type
    Params:     type
    Presets:    dict
    registries: dict | None = None
    sim_only:   bool | None = None

def resolve_preset(presets, name):
    p = presets[name]
    if "base" in p:
        base = resolve_preset(presets, p["base"])
        return {**base, **{k: v for k, v in p.items() if k != "base"}}
    return p

def build_from_registry(registry, spec_string):
    type_, preset = spec_string.split(":")
    try:
        spec = registry[type_]
    except KeyError:
        raise ValueError(f"Unknown type: {type_}")

    raw = resolve_preset(spec.Presets, preset)

    resolved = {}
    for k, v in raw.items():
        sub_registry = (spec.registries or {}).get(k)
        if isinstance(v, str) and ":" in v:
            resolved[k] = build_from_registry(sub_registry, v)
        elif isinstance(v, dict) and sub_registry:
            resolved[k] = {name: build_from_registry(sub_registry, s) for name, s in v.items()}
        else:
            resolved[k] = v

    return spec.cls(spec.Params(**resolved))
```

Everyone imports from `src.shared`. No circular imports, one source of truth.

---

## Import order (to avoid circular imports)

Build bottom-up. Each layer only imports from layers below it.

```
src/shared.py               ← no project imports
src/system/plant/           ← imports shared
src/system/controller/      ← imports shared (PlantParams)
src/system/estimator/       ← imports shared (PlantParams)
src/system/sensor/          ← imports shared
src/system/actuator/        ← imports shared
src/system/supervisor/      ← imports shared
src/system/__init__.py      ← imports all of the above
src/experiment/logger/      ← imports shared
src/experiment/stop_condition/ ← imports shared
src/experiment/visualizer/  ← imports shared
src/experiment/pacing/      ← imports shared
src/experiment/scheduler/   ← imports shared
src/experiment/progress/    ← imports shared
src/experiment/__init__.py  ← imports system + all experiment subfolders
src/registries.py           ← imports everything (convenience only)
main.py                     ← imports registries, build_from_registry, CONFIG_PRESETS
```
