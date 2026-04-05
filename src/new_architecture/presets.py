import numpy as np







# ── Actuators ─────────────────────────────────────────────────

SERVO_PRESETS = {
    "default": {
        "port":      "/dev/ttyUSB1",
        "frequency": 250,
    }
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
        "L":         0.15,
        "fps":       24,
        "add_trail": True,
        "mech":      "five_bar:default",
    }
}