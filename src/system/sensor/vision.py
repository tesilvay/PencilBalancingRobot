from dataclasses import dataclass

import numpy as np
from core.sim_types import (
    SystemState,
    PoseMeasurement,
)


@dataclass
class VisionParams:
    interface: object
    algo:      object
    reg_model: object


VISION_PRESETS = {
    "sim_analytic": {
        "interface": "sim_analytic:default",
        "algo":      "hough:default",
        "reg_model": "none:default",
    },
    "sim_dvs_hough": {
        "interface": "sim_dvs:default",
        "algo":      "hough:default",
        "reg_model": "simple:default",
    },
    "sim_dvs_sam": {"base": "sim_dvs_hough", "algo": "sam:default"},
    "real_dvs":    {"base": "sim_dvs_hough",  "interface": "real_dvs:default"},
}


class Vision:
    def __init__(self, params: VisionParams):
        self.interface = params.interface
        self.algo = params.algo
        self.reg_model = params.reg_model
