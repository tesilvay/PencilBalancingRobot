from __future__ import annotations

from typing import Any

from UI.preset_capabilities import PRESET_CAPABILITIES


PRESET_ORDER = ["sim", "vision_real", "actuation_real", "real", "other"]
REAL_PRESETS = ["sim", "vision_real", "actuation_real", "real"]


def get_capability_preset(selected_preset: str, other_base_preset: str) -> str:
    if selected_preset == "other":
        return other_base_preset
    return selected_preset


def get_defaults_for_preset(preset: str) -> dict[str, Any]:
    if preset not in PRESET_CAPABILITIES:
        return {}
    return dict(PRESET_CAPABILITIES[preset].get("defaults", {}))


def reset_selections_for_preset(preset: str) -> dict[str, Any]:
    return get_defaults_for_preset(preset)
