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
      "controller": ["lqr", "pole"],
      "estimator": ["kalman", "lpf", "fde"],
    },
  },
  "vision_real": {
    "defaults": {
      "experiment": "single",
      "actuator": "sim",
      "controller": "lqr",
      "estimator": "kalman",
    },
    "allowed": {
        "controller": ["lqr", "pole"],
        "estimator": ["kalman", "lpf", "fde"],
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
        "controller": ["lqr", "pole", "circle"],
        "estimator": ["kalman", "lpf", "fde"],
    },
  },
  "real": {
    "defaults": {
      "experiment": "single",
      "actuator": "servo",
      "controller": "lqr",
      "estimator": "kalman",
    },
    "allowed": {
        "actuator": ["servo", "mock"],
        "controller": ["lqr", "pole", "circle"],
        "estimator": ["kalman", "lpf", "fde"],
    },
  },
}