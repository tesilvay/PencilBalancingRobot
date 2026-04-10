from argparse import Namespace
import sys
import types
from pathlib import Path

import numpy as np


def _install_control_stub():
    if "control" in sys.modules:
        return

    control = types.ModuleType("control")

    def ss(A, B, C, D):
        return types.SimpleNamespace(A=np.array(A), B=np.array(B), C=np.array(C), D=np.array(D))

    def c2d(sys, dt):
        del dt
        return types.SimpleNamespace(A=np.array(sys.A), B=np.array(sys.B))

    def place(A, B, poles):
        del poles
        return np.zeros((np.array(B).shape[1], np.array(A).shape[0]))

    def lqr(A, B, Q, R):
        del Q, R
        return np.zeros((np.array(B).shape[1], np.array(A).shape[0])), None, None

    def ctrb(A, B):
        A = np.array(A)
        B = np.array(B)
        n = A.shape[0]
        return np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])

    def obsv(A, C):
        A = np.array(A)
        C = np.array(C)
        n = A.shape[0]
        return np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(n)])

    control.ss = ss
    control.c2d = c2d
    control.place = place
    control.lqr = lqr
    control.ctrb = ctrb
    control.obsv = obsv
    sys.modules["control"] = control

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_install_control_stub()

from main import build_experiment, collect_overrides


def _args(**updates):
    base = {
        "preset": "sim",
        "system": None,
        "logger": None,
        "stop_condition": None,
        "realtime_visualizer": None,
        "offline_visualizer": None,
        "progress": None,
        "plants": None,
        "controllers": None,
        "estimators": None,
        "sensor": None,
        "actuator": None,
        "supervisor": None,
        "set": [],
        "list": False,
        "graph": None,
    }
    base.update(updates)
    return Namespace(**base)


def test_collect_overrides_splits_experiment_and_system_paths():
    args = _args(
        estimators="lpf:test,kalman:test",
        sensor="sim_dvs:hough",
        set=["n_trials=3", "system.workspace.safe_radius=0.05"],
    )

    overrides = collect_overrides(args)

    assert overrides["experiment"] == [(["n_trials"], "3")]
    assert overrides["system"] == [
        (["estimators"], "lpf:test,kalman:test"),
        (["sensor"], "sim_dvs:hough"),
        (["workspace", "safe_radius"], "0.05"),
    ]


def test_build_experiment_overrides_system_estimators_list():
    overrides = {
        "experiment": [],
        "system": [(["estimators"], "lpf:test,kalman:test")],
    }

    experiment = build_experiment("default:sim", overrides)

    assert len(experiment.system.estimators) == 2
    assert type(experiment.system.estimators[0]).__name__ == "LowPassFiniteDifferenceEstimator"
    assert type(experiment.system.estimators[1]).__name__ == "KalmanEstimator"


def test_build_experiment_overrides_system_nested_dataclass_field():
    overrides = {
        "experiment": [],
        "system": [(["workspace", "safe_radius"], "0.05")],
    }

    experiment = build_experiment("default:sim", overrides)

    assert experiment.system.workspace.safe_radius == 0.05


def test_build_experiment_overrides_experiment_nested_dataclass_field():
    overrides = {
        "experiment": [(["timing", "dt"], "0.01")],
        "system": [],
    }

    experiment = build_experiment("default:sim", overrides)

    assert experiment.dt == 0.01
