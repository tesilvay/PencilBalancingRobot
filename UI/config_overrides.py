from __future__ import annotations

from dataclasses import fields, is_dataclass
import types
from typing import Any, Union, get_args, get_origin, get_type_hints


class OverrideError(ValueError):
    """Raised when an override path or value is invalid."""


def _strip_optional(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin is None:
        return tp
    args = get_args(tp)
    if origin in (Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return tp


def _coerce_value(value: Any, expected_type: Any, path: str) -> Any:
    expected_type = _strip_optional(expected_type)
    if expected_type is Any:
        return value

    # Dataclass replacement is not supported; use dot-path leaf updates instead.
    if is_dataclass(expected_type):
        raise OverrideError(
            f"Cannot replace dataclass field at '{path}'. "
            "Override nested leaf fields (e.g. 'params.run.dt')."
        )

    try:
        if expected_type is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                v = value.strip().lower()
                if v in {"1", "true", "yes", "on"}:
                    return True
                if v in {"0", "false", "no", "off"}:
                    return False
            raise OverrideError(
                f"Cannot coerce value '{value}' to bool at '{path}'"
            )

        if expected_type in (int, float, str):
            return expected_type(value)

        origin = get_origin(expected_type)
        if origin in (list, tuple):
            if not isinstance(value, (list, tuple)):
                raise OverrideError(
                    f"Expected sequence for '{path}', got {type(value).__name__}"
                )
            elem_type = get_args(expected_type)[0] if get_args(expected_type) else Any
            coerced = [_coerce_value(v, elem_type, f"{path}[]") for v in value]
            return tuple(coerced) if origin is tuple else coerced
    except OverrideError:
        raise
    except Exception as exc:
        raise OverrideError(
            f"Failed to coerce override '{path}'={value!r} to {expected_type}"
        ) from exc

    return value


def _resolve_path(root: Any, path: str) -> tuple[Any, str]:
    parts = [p for p in path.split(".") if p]
    if not parts:
        raise OverrideError("Override path cannot be empty")

    target = root
    for i, part in enumerate(parts[:-1]):
        if isinstance(target, dict):
            if part not in target:
                raise OverrideError(
                    f"Unknown key '{part}' while resolving '{path}'"
                )
            target = target[part]
            continue

        if not hasattr(target, part):
            prefix = ".".join(parts[: i + 1])
            raise OverrideError(f"Unknown attribute '{prefix}' in '{path}'")
        target = getattr(target, part)

    return target, parts[-1]


def _field_type_for_attr(parent: Any, attr: str) -> Any:
    if not is_dataclass(parent):
        return Any
    # get_type_hints resolves forward refs reliably
    hints = get_type_hints(type(parent))
    if attr in hints:
        return hints[attr]
    for f in fields(parent):
        if f.name == attr:
            return f.type
    return Any


def apply_overrides(root: Any, overrides: dict[str, Any], _prefix: str = "") -> Any:
    """
    Apply dot-path overrides to a nested dataclass/dict object in-place.

    Example
    -------
    overrides = {
      "params.run.dt": 0.002,
      "params.hardware.dvs_hough.mixing_factor": 0.03,
      "default_variant.noise_std": 0.02,
    }
    apply_overrides(setup, overrides)
    """
    if not isinstance(overrides, dict):
        raise OverrideError("Overrides must be a dict of {path: value}")

    for raw_path, value in overrides.items():
        path = f"{_prefix}{raw_path}" if _prefix else raw_path
        parent, key = _resolve_path(root, path)

        if isinstance(parent, dict):
            if key not in parent:
                raise OverrideError(f"Unknown key '{path}'")
            parent[key] = value
            continue

        if not hasattr(parent, key):
            raise OverrideError(f"Unknown attribute '{path}'")

        expected_type = _field_type_for_attr(parent, key)
        coerced = _coerce_value(value, expected_type, path)
        setattr(parent, key, coerced)

    return root
