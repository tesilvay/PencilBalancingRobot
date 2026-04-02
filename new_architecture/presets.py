import numpy as np

TIMING_PRESETS = {"default": {"total_time": 5.0, "dt": 4e-3}}

# Shared for any class that needs no preset fields (null / noop / empty).
NULL_PRESETS = {"default": {}}


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


# ── Estimators ────────────────────────────────────────────────

LPF_PRESETS = {"default": {"alpha": 0.93}}

KALMAN_PRESETS = {
    "default": {
        "plant":      "default:default",
        "timing":     "default:default",
        "q_pose_pos": 1e-6,
        "q_pose_ang": 1e-6,
        "q_vel_pos":  1e-3,
        "q_vel_ang":  1e-2,
        "r_pose_pos": 1e-2,
        "r_pose_ang": 7e-2,
    }
}

FULL_KALMAN_PRESETS = {
    "default": {
        "plant":      "default:default",
        "timing":     "default:default",
        "q_pose_pos": 1e-8,
        "q_pose_ang": 1e-8,
        "q_vel_pos":  1e-4,
        "q_vel_ang":  1e-4,
        "r_pose_pos": 1e-7,
        "r_vel_pos":  1e-4,
        "r_pose_ang": 1e-4,
        "r_vel_ang":  1e-6,
        "lpf_alpha":  0.95,
    }
}


# ── Vision: Line Algorithms ──────────────────────────────────

HOUGH_PRESETS = {
    "default": {
        "mixing_factor":    0.02,
        "inlier_stddev_px": 4.0,
        "min_determinant":  1e-6,
    }
}

SAM_PRESETS = {
    "default": {
        "min_points": 50,
    }
}


# ── Vision: Regression Models ────────────────────────────────

SIMPLE_REG_PRESETS = {
    "default": {
        "calibration_path": "hardware/calibration_files/dvs_affine_calibration.json",
    }
}


# ── Vision: Interfaces ───────────────────────────────────────

SIM_ANALYTIC_PRESETS = {
    "default": {
        "noise_std":   None,
        "delay_steps": 0,
    },
    "noisy": {
        "base": "default",
        "noise_std":   1e-3,
        "delay_steps": 2,
    },
}

SIM_DVS_PRESETS = {
    "default": {
        "dvs_mask_line_y_cam1": 160,
        "dvs_mask_line_y_cam2": 190,
    }
}

REAL_DVS_PRESETS = {
    "default": {
        "cam1_device":              None,
        "cam2_device":              None,
        "dvs_mask_line_y_cam1":     160,
        "dvs_mask_line_y_cam2":     190,
        "noise_filter_duration_ms": None,
    }
}


# ── Vision: Composite ────────────────────────────────────────

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


# ── Actuators ─────────────────────────────────────────────────

SERVO_PRESETS = {
    "default": {
        "port":      "/dev/ttyUSB1",
        "frequency": 250,
    }
}


# ── Supervisors ──────────────────────────────────────────────

DYNAMIC_SUPERVISOR_PRESETS = {
    "default": {
        "stable_threshold":  0.035,
        "stable_hold_s":     2.0,
        "consistent_hold_s": 1.0,
        "loss_threshold":    0.3,
    }
}

STATIC_SUPERVISOR_PRESETS = {
    "default": {
        "controller_key": "smooth",
        "estimator_key":  "kalman",
    }
}


# ── System ────────────────────────────────────────────────────

SYSTEM_PRESETS = {
    "dynamic_sim": {
        "plant":       "sim:default",
        "controllers": {"follower": "pole:default", "smooth": "smooth_pole:default"},
        "estimators":  {"lpf": "lpf:default", "kalman": "kalman:default"},
        "vision":      "default:sim_dvs_hough",
        "actuator":    "mock:default",
        "supervisor":  "dynamic:default",
    },
    "simple_sim": {
        "base": "dynamic_sim",
        "controllers": {"smooth": "smooth_pole:default"},
        "estimators":  {"kalman": "kalman:default"},
        "supervisor":  "static:default",
    },
    "real": {
        "base": "dynamic_sim",
        "vision":   "default:real_dvs",
        "actuator": "servo:default",
    },
}


# ── Stop Conditions ──────────────────────────────────────────

FALL_CONDITION_PRESETS = {
    "default": {
        "max_angle_deg": 45.0,
    }
}

STABILIZED_CONDITION_PRESETS = {
    "default": {
        "tol_ang_deg": 10.0,
        "tol_m":       10e-3,
        "settle_time": 0.5,
    }
}

MAX_STEPS_CONDITION_PRESETS = {
    "default": {
        "timing":      "default:default",
        "tol_ang_deg": 10.0,
        "tol_m":       10e-3,
        "settle_time": 0.5,
    }
}

ANY_STOP_CONDITION_PRESETS = {
    "default": {
        "conditions": {
            "fall":      "fall:default",
            "max_steps": "max_steps:default",
        }
    },
    "early_stop": {
        "conditions": {
            "fall":       "fall:default",
            "stabilized": "stabilized:default",
            "max_steps":  "max_steps:default",
        }
    },
}


# ── Visualizers ───────────────────────────────────────────────

SIM_DVS_VISUALIZER_PRESETS = {
    "default": {
        "width":  346,
        "height": 260,
    }
}

REAL_DVS_VISUALIZER_PRESETS = {
    "default": {
        "width":       346,
        "height":      260,
        "mask_y_cam1": 160,
        "mask_y_cam2": 190,
    }
}

ONE_DVS_VISUALIZER_PRESETS = {
    "default": {
        "cam_index":    0,
        "width":        346,
        "height":       260,
        "surface_gain": 50.0,
    }
}

SIM_DVS_WORKSPACE_VISUALIZER_PRESETS = {
    "default": {
        "width":  346,
        "height": 260,
    }
}

REAL_DVS_WORKSPACE_VISUALIZER_PRESETS = {
    "default": {
        "width":       346,
        "height":      260,
        "mask_y_cam1": 160,
        "mask_y_cam2": 190,
    }
}

VISUALIZER_3D_PRESETS = {
    "default": {
        "L":   0.15,
        "fps": 60,
    }
}


# ── Progress ──────────────────────────────────────────────────

PROGRESS_PRESETS = {
    "default": {
        "width": 30,
    }
}


# ── Pacing ────────────────────────────────────────────────────

REALTIME_PACING_PRESETS = {
    "default": {
        "timing": "default:default",
    }
}


# ── Scheduler ─────────────────────────────────────────────────

SCHEDULER_PRESETS = {
    "default": {
        "timing":             "default:default",
        "actuator_frequency": 250,
        "render_frequency":   30,
    }
}


# ── Experiment ────────────────────────────────────────────────

EXPERIMENT_PRESETS = {
    "sim": {
        "system":         "default:simple_sim",
        "logger":         "default:default",
        "stop_condition": "any:default",
        "visualizer":     {},
        "progress":       "default:default",
        "pacing":         "null:default",
        "scheduler":      "realtime:default",
    },
    "realtime_sim": {
        "base": "sim",
        "visualizer": {"sim": "sim:default"},
        "pacing":     "realtime:default",
    },
    "real": {
        "base": "sim",
        "system":     "default:real",
        "visualizer": {"real": "real:default"},
        "pacing":     "realtime:default",
    },
}