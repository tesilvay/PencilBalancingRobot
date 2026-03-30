PRESET_CAPABILITIES = {
  "sim": {
    "defaults": {
      "experiment": "single",
      "actuator": "sim",
      "controller": "lqr",
      "estimator": "kalman",
    },
    "allowed": {
      "experiment": ["single", "montecarlo", "benchmark", "sweep"],
      "controller": ["lqr", "pole", "smooth_pole", "smooth_lqr"],
      "estimator": ["kalman", "lpf", "fde", "kalman_full"],
    },
  },
  "vision_real": {
    "defaults": {
      "experiment": "single",
      "actuator": "mock",
      "controller": "lqr",
      "estimator": "kalman",
    },
    "allowed": {
        "controller": ["lqr", "pole", "smooth_pole", "smooth_lqr"],
        "estimator": ["kalman", "lpf", "fde", "kalman_full"],
    },
  },
  "actuation_real": {
    "defaults": {
      "experiment": "single",
      "actuator": "servo",
      "controller": "lqr",
      "estimator": "kalman",
    },
    "allowed": {
        "actuator": ["servo", "mock"],
        "controller": ["lqr", "pole", "smooth_pole", "smooth_lqr", "circle"],
        "estimator": ["kalman", "lpf", "fde", "kalman_full"],
    },
  },
  "real": {
    "defaults": {
      "experiment": "single",
      "actuator": "servo",
      "controller": "smooth_pole",
      "estimator": "kalman_full",
    },
    "allowed": {
        "actuator": ["servo", "mock"],
        "controller": ["lqr", "pole", "smooth_pole", "smooth_lqr", "circle"],
        "estimator": ["kalman", "lpf", "fde", "kalman_full"],
    },
  },
}