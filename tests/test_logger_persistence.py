import pickle
from pathlib import Path
import sys
import types

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

from src.experiment.logger.logger import Logger, LoggerParams
from src.shared import ControlInput, State, StepData, TableAccel


def _state(*, px: float = 0.0, py: float = 0.0) -> State:
    return State(px=px, vx=0.0, ax=0.0, wx=0.0, py=py, vy=0.0, ay=0.0, wy=0.0)


def _step(state_name: str | None, *, px: float = 0.0, py: float = 0.0) -> StepData:
    return StepData(
        x=_state(px=px, py=py),
        u=ControlInput(px_cmd=px, py_cmd=py),
        acc=TableAccel(0.0, 0.0),
        x_hat=_state(px=px, py=py),
        innovation=np.zeros(4),
        mech_joints=np.full((3, 2), np.nan),
        offset_xy=np.zeros(2),
        offset_latched=True,
        supervisor_state=state_name,
    )


def test_logger_saves_full_trial_for_static_runs(tmp_path: Path):
    logger = Logger(LoggerParams(save_dir=str(tmp_path)))
    logger.reset(_step("STATIC", px=0.0, py=0.0))
    logger.record(_step("STATIC", px=0.01, py=-0.02))
    logger.record(_step("STATIC", px=0.02, py=-0.01))

    logger.flush_pending_chunks()

    files = sorted(tmp_path.glob("logger_chunk_*.pkl"))
    assert len(files) == 1

    with files[0].open("rb") as fh:
        payload = pickle.load(fh)

    assert payload["reason"] == "full_trial"
    assert payload["start_index"] == 0
    assert payload["stop_index"] == 3
    result = payload["result"]
    assert result.state_history.shape == (3, 8)
    np.testing.assert_allclose(result.cmd_history[-1], [0.02, -0.01], atol=1e-12)


def test_logger_does_not_save_full_trial_for_real_pre_acquisition_runs(tmp_path: Path):
    logger = Logger(LoggerParams(save_dir=str(tmp_path)))
    logger.reset(_step("SERVO_CENTERING"))
    logger.record(_step("SERVO_CENTERING", px=0.01))
    logger.record(_step("SERVO_CENTERING", px=0.02))

    logger.flush_pending_chunks()

    files = sorted(tmp_path.glob("logger_chunk_*.pkl"))
    assert files == []
