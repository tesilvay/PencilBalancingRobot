"""Stub composite_layout module — real implementation lives outside this repo."""
import numpy as np


def get_default_window_size(
    *,
    has_cams: bool = True,
    has_workspace: bool = True,
    single_cam: bool = False,
    one_dvs_side_panel: bool = False,
) -> tuple[int, int]:
    return 1200, 600


def build_composite(*args, **kwargs) -> np.ndarray:
    return np.zeros((600, 1200, 3), dtype=np.uint8)


def build_one_dvs_composite(*args, **kwargs) -> np.ndarray:
    return np.zeros((600, 1200, 3), dtype=np.uint8)
