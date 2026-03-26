from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from UI.preset_capabilities import PRESET_CAPABILITIES


def resolve_ui_config(preset: str, selections: dict[str, Any]) -> dict[str, Any]:
    if preset not in PRESET_CAPABILITIES:
        raise ValueError(f"Unknown preset: {preset}")

    cap = PRESET_CAPABILITIES[preset]
    defaults = dict(cap.get("defaults", {}))
    allowed = cap.get("allowed", {})
    resolved = dict(defaults)

    for key, value in (selections or {}).items():
        if key not in allowed:
            continue
        options = allowed[key]
        if value in options:
            resolved[key] = value

    # Keep radius available for circle controller.
    if "radius" in selections:
        resolved["radius"] = selections["radius"]

    return resolved


def build_backend_args(preset: str, resolved: dict[str, Any]) -> argparse.Namespace:
    args = argparse.Namespace(
        preset=preset,
        experiment=resolved.get("experiment", "single"),
        actuator=resolved.get("actuator"),
        trials=resolved.get("trials"),
        controller=resolved.get("controller"),
        estimator=resolved.get("estimator"),
        radius=resolved.get("radius"),
        graph=False,
    )
    return args


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_cli_command(
    preset: str,
    resolved: dict[str, Any],
    dev_overrides: dict[str, Any] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "main.py",
        "--preset",
        str(preset),
        "--experiment",
        str(resolved.get("experiment", "single")),
    ]

    if resolved.get("actuator") is not None:
        command.extend(["--actuator", str(resolved["actuator"])])
    if resolved.get("controller") is not None:
        command.extend(["--controller", str(resolved["controller"])])
    if resolved.get("estimator") is not None:
        command.extend(["--estimator", str(resolved["estimator"])])
    if resolved.get("radius") is not None:
        command.extend(["--radius", str(resolved["radius"])])
    if resolved.get("trials") is not None:
        command.extend(["--trials", str(resolved["trials"])])
    if dev_overrides:
        command.extend(["--overrides-json", json.dumps(dev_overrides)])

    return command


def run_from_ui(
    preset: str,
    resolved: dict[str, Any],
    dev_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    command = build_cli_command(preset, resolved, dev_overrides)
    repo = _repo_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(repo)

    try:
        proc = subprocess.run(
            command,
            cwd=str(repo),
            env=env,
            capture_output=True,
            text=True,
        )
        logs = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        if proc.returncode == 0:
            return {"ok": True, "logs": logs, "command": " ".join(command)}
        return {
            "ok": False,
            "error": f"Backend exited with code {proc.returncode}",
            "logs": logs,
            "command": " ".join(command),
        }
    except Exception as exc:  # broad catch to keep UI alive
        return {"ok": False, "error": f"Failed to launch subprocess: {exc}", "logs": "", "command": " ".join(command)}
