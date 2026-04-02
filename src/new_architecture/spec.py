from dataclasses import dataclass
from typing import Any


@dataclass
class Spec:
    cls: type
    Params: type
    Presets: dict
    registries: dict[str, Any] | None = None
    sim_only: bool | None = None
