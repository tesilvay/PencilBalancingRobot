from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from perception.estimator_diagnostics import EstimatorDiagnosticSnapshot


@dataclass
class DiagnosticsConfig:
    enabled: bool = False
    terminal_hz: float = 2.0
    terminal: bool = True
    keep_last: int = 0


class DiagnosticsManager:
    def __init__(self, config: DiagnosticsConfig):
        self._config = config
        self._last_emit_wall_s = float("-inf")
        self._history: deque[EstimatorDiagnosticSnapshot] | None = (
            deque(maxlen=config.keep_last) if config.keep_last > 0 else None
        )

    @classmethod
    def from_run_params(cls, run) -> DiagnosticsManager | None:
        if not getattr(run, "estimator_diagnostics_enabled", False):
            return None
        cfg = DiagnosticsConfig(
            enabled=True,
            terminal_hz=float(
                getattr(run, "estimator_diagnostics_terminal_hz", 2.0)
            ),
            terminal=getattr(run, "estimator_diagnostics_terminal", True),
            keep_last=int(getattr(run, "estimator_diagnostics_history", 0)),
        )
        return cls(cfg)

    def emit(self, snapshot: EstimatorDiagnosticSnapshot | None) -> None:
        if not self._config.enabled or snapshot is None:
            return
        if self._history is not None:
            self._history.append(snapshot)
        if not self._config.terminal:
            return
        if not self._should_emit_terminal():
            return
        print(snapshot.to_terminal_line(), flush=True)

    def _should_emit_terminal(self) -> bool:
        min_dt = 1.0 / max(self._config.terminal_hz, 1e-6)
        now = time.perf_counter()
        if now - self._last_emit_wall_s >= min_dt:
            self._last_emit_wall_s = now
            return True
        return False

    def get_history(self) -> list[EstimatorDiagnosticSnapshot]:
        if self._history is None:
            return []
        return list(self._history)
