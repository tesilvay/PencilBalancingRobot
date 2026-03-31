from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from perception.kalman_core import KalmanStepResult


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a))))


def _kalman_velocity_gain_summary(K: np.ndarray) -> tuple[float, float]:
    if K.shape[0] < 8:
        return float("nan"), float("nan")
    k_pos = float(
        np.hypot(np.linalg.norm(K[1, :]), np.linalg.norm(K[5, :]))
    )
    k_ang = float(
        np.hypot(np.linalg.norm(K[3, :]), np.linalg.norm(K[7, :]))
    )
    return k_pos, k_ang


@dataclass
class EstimatorDiagnosticSnapshot:
    estimator_name: str
    step_idx: int
    t_s: float
    dt_s: float
    measurement_fresh: bool
    z_changed: bool
    diag_P: np.ndarray | None = None
    innovation_rms: float | None = None
    nis: float | None = None
    K_vel_pos_norm: float | None = None
    K_vel_ang_norm: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_terminal_line(self) -> str:
        parts = [
            f"[{self.estimator_name}] t={self.t_s:.3f}s step={self.step_idx}",
            f"fresh_z={self.measurement_fresh}",
            f"z_changed={self.z_changed}",
        ]
        if self.innovation_rms is not None:
            parts.append(f"|y|_rms={self.innovation_rms:.4g}")
        if self.nis is not None:
            parts.append(f"NIS={self.nis:.3f}")
        if self.diag_P is not None:
            p = self.diag_P
            parts.append(
                f"P_diag[max vel]={max(float(p[1]), float(p[5])):.2e} "
                f"[max pos]={max(float(p[0]), float(p[4])):.2e}"
            )
        if self.K_vel_pos_norm is not None and np.isfinite(self.K_vel_pos_norm):
            parts.append(f"||K_vel_pos||={self.K_vel_pos_norm:.4g}")
        if self.K_vel_ang_norm is not None and np.isfinite(self.K_vel_ang_norm):
            parts.append(f"||K_vel_ang||={self.K_vel_ang_norm:.4g}")
        return " ".join(parts)


def build_kalman_snapshot(
    *,
    estimator_name: str,
    step_idx: int,
    t_s: float,
    dt_s: float,
    measurement_fresh: bool,
    z_changed: bool,
    step: KalmanStepResult,
) -> EstimatorDiagnosticSnapshot:
    y = step.y
    k_pos, k_ang = _kalman_velocity_gain_summary(step.K)
    return EstimatorDiagnosticSnapshot(
        estimator_name=estimator_name,
        step_idx=step_idx,
        t_s=t_s,
        dt_s=dt_s,
        measurement_fresh=measurement_fresh,
        z_changed=z_changed,
        diag_P=np.diag(step.P).copy(),
        innovation_rms=_rms(y),
        nis=step.nis,
        K_vel_pos_norm=k_pos,
        K_vel_ang_norm=k_ang,
        extra={"cond_S": float(np.linalg.cond(step.S))},
    )
