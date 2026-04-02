import numpy as np

TIMING_PRESETS = {"default": {"total_time": 5.0, "dt": 4e-3}}


PLANT_PRESETS = {
    "default": {
        "g"             : 9.81,
        "l"             : 0.15,
        "tau"           : 0.03,
        "zeta"          : 0.8,
        "max_acc"       : 9.81 * 3,
        "num_states"    : 8,
        "x_ref"         : 0.0,
        "y_ref"         : 0.0,
        "safe_radius"   : 68e-3
    }
}

POLE_PRESETS = {
    "default": {
        "plant": "default:default",          # controller owns its plant reference
        "poles": [-14, -16, -18, -20] * 2,
    }
}
    
    
LQR_PRESETS = {
    "default": {
        "plant": "default:default",
        "Q_single_axis": np.diag([0.01, 0.01, 100, 10]),  # x, x_dot, alpha, alpha_dot
        "R":             np.eye(2) * 1e6,
    }
}


SMOOTH_POLE_PRESETS = {
    "default": {
        "plant":         "default:default",
        "timing":        "default:default",
        "s_poles":       [-14, -16, -18, -20] * 2,
        "slew_poles":    0.95 # 0 same as without it, 1 u frozen
    }
}

SMOOTH_LQR_PRESETS = {
    "default": {
        "plant":            "default:default",
        "Q_single_axis":    np.diag([0.01, 0.01, 100, 10]),  # x, x_dot, alpha, alpha_dot
        "R":                np.eye(2) * 1e6,
        "q_u":              1e-6,
        "r_delta":          1e4,
    }
}
    
CIRCLE_PRESETS = {
    "default": {
        "plant":         "default:default",
        "timing":        "default:default",
        "period_s":       18,
    }
}



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