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
class NullParams:
    pass


NULL_PRESETS = {"default": {}}

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


