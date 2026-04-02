from dataclasses import dataclass
import numpy as np

@dataclass
class TimingParams:
    total_time: float
    dt:         float

@dataclass
class PlantParams:
    g: float
    l: float
    tau: float
    zeta: float
    max_acc: float | None = None
    num_states: int
    x_ref: float
    y_ref: float
    safe_radius: float | None = None

@dataclass
class PoleParams:
    plant:  PlantParams   # physical constants live here, no copying
    poles:  list[float]
    
    
@dataclass
class LQRParams:
    plant:          PlantParams   # physical constants live here, no copying
    Q_single_axis:  np.ndarray
    R:              np.ndarray 


@dataclass
class SmoothPoleParams:
    plant:          PlantParams   # physical constants live here, no copying
    timing:         TimingParams
    s_poles:        list[float]
    slew_poles:     float 
    
@dataclass
class SmoothLQRParams:
    plant:          PlantParams   # physical constants live here, no copying
    Q_single_axis:  np.ndarray
    q_u:            float
    r_delta:        float
    
@dataclass
class CircleParams:
    plant:          PlantParams   # physical constants live here, no copying
    timing:         TimingParams
    period_s:       float

