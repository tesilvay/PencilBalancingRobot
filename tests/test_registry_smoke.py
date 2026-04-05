"""
Smoke test: build_from_registry succeeds for every preset in each src registry.

Run with:
    python -m pytest tests/test_registry_smoke.py -v
"""

import pytest

from src.shared import build_from_registry

from src.system.controller import CONTROLLER_REGISTRY
from src.system.estimator import ESTIMATOR_REGISTRY
from src.system.plant import PLANT_REGISTRY
from src.system.actuator import ACTUATOR_REGISTRY
from src.system.supervisor import SUPERVISOR_REGISTRY
from src.system.sensor import SENSOR_REGISTRY
from src.system.sensor.algo import LINE_ALGO_REGISTRY
from src.system.sensor.observation_model import REG_MODEL_REGISTRY
from src.experiment.logger import LOGGER_REGISTRY
from src.experiment.stop_condition import STOP_CONDITION_REGISTRY
from src.experiment.progress import PROGRESS_REGISTRY
from src.experiment.pacing import PACING_REGISTRY
from src.experiment.scheduler import SCHEDULER_REGISTRY
from src.experiment.realtime_visualizer import REALTIME_VISUALIZER_REGISTRY
from src.experiment.offline_visualizer import OFFLINE_VISUALIZER_REGISTRY
from src.system import SYSTEM_REGISTRY
from src.experiment import EXPERIMENT_REGISTRY


def _collect_cases(registry, registry_name, *, skip_types=(), xfail_types=()):
    """Yield (registry, spec_string, marks) for every type:preset pair."""
    for type_key, spec in registry.items():
        for preset_name in spec.Presets:
            spec_str = f"{type_key}:{preset_name}"
            marks = []
            if type_key in skip_types:
                marks.append(pytest.mark.skip(reason=f"{type_key} needs external resources"))
            elif type_key in xfail_types:
                marks.append(pytest.mark.xfail(reason=f"{type_key} may need runtime deps", strict=False))
            yield pytest.param(registry, spec_str, id=f"{registry_name}/{spec_str}", marks=marks)


_LEAF_CASES = [
    *_collect_cases(CONTROLLER_REGISTRY, "controller"),
    *_collect_cases(ESTIMATOR_REGISTRY, "estimator"),
    *_collect_cases(PLANT_REGISTRY, "plant"),
    *_collect_cases(ACTUATOR_REGISTRY, "actuator", skip_types=("servo",)),
    *_collect_cases(SUPERVISOR_REGISTRY, "supervisor"),
    *_collect_cases(LINE_ALGO_REGISTRY, "line_algo"),
    *_collect_cases(REG_MODEL_REGISTRY, "reg_model", skip_types=("simple",)),
    *_collect_cases(LOGGER_REGISTRY, "logger"),
    *_collect_cases(STOP_CONDITION_REGISTRY, "stop_condition"),
    *_collect_cases(PROGRESS_REGISTRY, "progress"),
    *_collect_cases(PACING_REGISTRY, "pacing"),
    *_collect_cases(SCHEDULER_REGISTRY, "scheduler"),
]


@pytest.mark.parametrize("registry,spec_string", _LEAF_CASES)
def test_leaf_registry_build(registry, spec_string):
    obj = build_from_registry(registry, spec_string)
    assert obj is not None


_REALTIME_VIZ_CASES = list(
    _collect_cases(
        REALTIME_VISUALIZER_REGISTRY,
        "realtime_visualizer",
    )
)


@pytest.mark.parametrize("registry,spec_string", _REALTIME_VIZ_CASES)
def test_realtime_visualizer_registry_build(registry, spec_string):
    obj = build_from_registry(registry, spec_string)
    assert obj is not None


_OFFLINE_VIZ_CASES = list(
    _collect_cases(
        OFFLINE_VISUALIZER_REGISTRY,
        "offline_visualizer",
        xfail_types=("3d",),
    )
)


@pytest.mark.parametrize("registry,spec_string", _OFFLINE_VIZ_CASES)
def test_offline_visualizer_registry_build(registry, spec_string):
    obj = build_from_registry(registry, spec_string)
    assert obj is not None


_SENSOR_CASES = list(_collect_cases(
    SENSOR_REGISTRY, "sensor",
    skip_types=("real_dvs",),
))


@pytest.mark.parametrize("registry,spec_string", _SENSOR_CASES)
def test_sensor_registry_build(registry, spec_string):
    obj = build_from_registry(registry, spec_string)
    assert obj is not None


_SYSTEM_CASES = list(_collect_cases(
    SYSTEM_REGISTRY, "system",
    xfail_types=("default",),
))


@pytest.mark.parametrize("registry,spec_string", _SYSTEM_CASES)
def test_system_registry_build(registry, spec_string):
    obj = build_from_registry(registry, spec_string)
    assert obj is not None


_EXPERIMENT_CASES = list(_collect_cases(
    EXPERIMENT_REGISTRY, "experiment",
    xfail_types=("default",),
))


@pytest.mark.parametrize("registry,spec_string", _EXPERIMENT_CASES)
def test_experiment_registry_build(registry, spec_string):
    obj = build_from_registry(registry, spec_string)
    assert obj is not None
