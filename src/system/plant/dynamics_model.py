import numpy as np
from src.shared import PlantParams
import control as ct


def BuildAccModel(param: PlantParams):
    """
    Build the acceleration-input inverted-pendulum model for one axis and 2D.

    This exposes three closely related models:

    1. Physical coordinates, one axis:
       z = [x, x_dot, alpha, alpha_dot]^T
       u = x_ddot

    2. Controller-normal-form coordinates, one axis:
       xc = [x - l*alpha, x_dot - l*alpha_dot, -g*alpha, -g*alpha_dot]^T

    3. Augmented controller model, one axis:
       xa = [integral_error, xc]^T
       with integral_error_dot = r - xc[0]

    Returns a dictionary so the caller can inspect whichever form is useful.
    """
    p = param
    g = p.g
    l = p.com_length

    # Physical one-axis plant:
    #   x_ddot = u
    #   alpha_ddot = (g/l) * alpha - (1/l) * u
    A_axis = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g / l, 0.0],
    ])
    B_axis = np.array([
        [0.0],
        [1.0],
        [0.0],
        [-1.0 / l],
    ])

    # State transform into the same coordinate family used by the Triamec
    # controller derivation:
    #   xc1 = x - l * alpha
    #   xc2 = x_dot - l * alpha_dot
    #   xc3 = -g * alpha
    #   xc4 = -g * alpha_dot
    T_axis = np.array([
        [1.0, 0.0, -l, 0.0],
        [0.0, 1.0, 0.0, -l],
        [0.0, 0.0, -g, 0.0],
        [0.0, 0.0, 0.0, -g],
    ])

    T_axis_inv = np.linalg.inv(T_axis)
    A_ctrl_axis = T_axis @ A_axis @ T_axis_inv
    B_ctrl_axis = T_axis @ B_axis

    # Augment with an integral-of-position-error state:
    #   e_int_dot = r - xc1
    # For regulation, r enters as an external reference input, so it does not
    # appear in A/B here.
    A_aug_axis = np.block([
        [np.array([[0.0, -1.0, 0.0, 0.0, 0.0]])],
        [np.hstack([np.zeros((4, 1)), A_ctrl_axis])],
    ])
    B_aug_axis = np.vstack([np.zeros((1, 1)), B_ctrl_axis])

    # Build the full decoupled 2D system by repeating the one-axis model.
    Z4 = np.zeros((4, 4))
    Z4x1 = np.zeros((4, 1))
    A = np.block([
        [A_axis, Z4],
        [Z4, A_axis],
    ])
    B = np.block([
        [B_axis, Z4x1],
        [Z4x1, B_axis],
    ])

    T = np.block([
        [T_axis, Z4],
        [Z4, T_axis],
    ])
    A_ctrl = np.block([
        [A_ctrl_axis, Z4],
        [Z4, A_ctrl_axis],
    ])
    B_ctrl = np.block([
        [B_ctrl_axis, Z4x1],
        [Z4x1, B_ctrl_axis],
    ])

    Z5 = np.zeros((5, 5))
    Z5x1 = np.zeros((5, 1))
    A_aug = np.block([
        [A_aug_axis, Z5],
        [Z5, A_aug_axis],
    ])
    B_aug = np.block([
        [B_aug_axis, Z5x1],
        [Z5x1, B_aug_axis],
    ])

    return {
        "A_axis": A_axis,
        "B_axis": B_axis,
        "T_axis": T_axis,
        "A_ctrl_axis": A_ctrl_axis,
        "B_ctrl_axis": B_ctrl_axis,
        "A_aug_axis": A_aug_axis,
        "B_aug_axis": B_aug_axis,
        "A": A,
        "B": B,
        "T": T,
        "A_ctrl": A_ctrl,
        "B_ctrl": B_ctrl,
        "A_aug": A_aug,
        "B_aug": B_aug,
    }


def BuildAccModelWithLag(param: PlantParams):
    """
    Build a lag-aware position-command model for one axis and 2D.

    One-axis physical state:
      z = [x, x_dot, alpha, alpha_dot]^T
      u = x_cmd

    with
      x_ddot = (1/tau^2) * (u - x) - (2*zeta/tau) * x_dot
      alpha_ddot = (g/l) * alpha - (1/l) * x_ddot

    The controller-coordinate state keeps the same interpretation as the
    ideal acceleration-input controller family:
      xc = [x - l*alpha, x_dot - l*alpha_dot, -g*alpha, -g*alpha_dot]^T
    """
    p = param
    g = p.g
    l = p.com_length
    tau = p.tau
    zeta = p.zeta

    A_axis = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-1.0 / tau**2, -2.0 * zeta / tau, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0 / (l * tau**2), 2.0 * zeta / (l * tau), g / l, 0.0],
    ])
    B_axis = np.array([
        [0.0],
        [1.0 / tau**2],
        [0.0],
        [-1.0 / (l * tau**2)],
    ])

    T_axis = np.array([
        [1.0, 0.0, -l, 0.0],
        [0.0, 1.0, 0.0, -l],
        [0.0, 0.0, -g, 0.0],
        [0.0, 0.0, 0.0, -g],
    ])

    T_axis_inv = np.linalg.inv(T_axis)
    A_ctrl_axis = T_axis @ A_axis @ T_axis_inv
    B_ctrl_axis = T_axis @ B_axis

    Z4 = np.zeros((4, 4))
    Z4x1 = np.zeros((4, 1))
    A = np.block([
        [A_axis, Z4],
        [Z4, A_axis],
    ])
    B = np.block([
        [B_axis, Z4x1],
        [Z4x1, B_axis],
    ])

    T = np.block([
        [T_axis, Z4],
        [Z4, T_axis],
    ])
    A_ctrl = np.block([
        [A_ctrl_axis, Z4],
        [Z4, A_ctrl_axis],
    ])
    B_ctrl = np.block([
        [B_ctrl_axis, Z4x1],
        [Z4x1, B_ctrl_axis],
    ])

    return {
        "A_axis": A_axis,
        "B_axis": B_axis,
        "T_axis": T_axis,
        "A_ctrl_axis": A_ctrl_axis,
        "B_ctrl_axis": B_ctrl_axis,
        "A": A,
        "B": B,
        "T": T,
        "A_ctrl": A_ctrl,
        "B_ctrl": B_ctrl,
    }

def BuildLinearModel(param: PlantParams):
    p = param
    g = p.g
    l = p.com_length
    tau = p.tau
    zeta = p.zeta
    
    # State Space Representation
    # The x and y axes have the same dynamics
    # If we assume them to be independent (there can be coupling issues but they should be negligible)
    # We will build A and B matrices for one axis, and then reuse it for the other axis.=


    # Build A and B matrices (x axis)

    A_x = np.array([
        [0, 1, 0, 0],
        [-1/tau**2, -2*zeta/tau, 0, 0],
        [0, 0, 0, 1],
        [1/(l*tau**2), 2*zeta/(l*tau), g/l, 0]
    ])

    B_x = np.array([
        [0],
        [1/tau**2],
        [0],
        [-1/(l*tau**2)]
    ])

    # Build full 2d system (8 STATES, 2 INPUTS)

    # Zero blocks
    Z4 = np.zeros((4, 4))
    Z4x1 = np.zeros((4, 1))

    # Full A (8x8)
    A = np.block([
        [A_x, Z4],
        [Z4,  A_x]
    ])

    # Full B (8x2)
    B = np.block([
        [B_x, Z4x1],
        [Z4x1, B_x]
    ])
    
    # This shows which states are actually observable
    C = np.array([
        [1,0,0,0,0,0,0,0],  # x
        [0,0,0,0,1,0,0,0],  # y
        [0,0,1,0,0,0,0,0],  # alpha_x
        [0,0,0,0,0,0,1,0],  # alpha_y
    ])
    
    
    # # Controllability / Observability
 
    Co = ct.ctrb(A, B)
    Ob = ct.obsv(A, C)

    controllable = np.linalg.matrix_rank(Co) == A.shape[0]
    observable  = np.linalg.matrix_rank(Ob) == A.shape[0]

    # print(f"Controllable: {controllable} (rank {np.linalg.matrix_rank(Co)}/{A.shape[0]})")
    # print(f"Observable : {observable} (rank {np.linalg.matrix_rank(Ob)}/{A.shape[0]})")
    
    return A, B

def BuildLinearModel_Rod(param: PlantParams):
    p = param
    g = p.g
    l = p.com_length
    tau = p.tau
    zeta = p.zeta
    
    # State Space Representation
    # The x and y axes have the same dynamics
    # If we assume them to be independent (there can be coupling issues but they should be negligible)
    # We will build A and B matrices for one axis, and then reuse it for the other axis.=


    # Build A and B matrices (x axis)

    A_x = np.array([
        [0, 1, 0, 0],
        [-1/tau**2, -2*zeta/tau, 0, 0],
        [0, 0, 0, 1],
        [3/(4*l*tau**2), 3*2*zeta/(4*l*tau), 3*g/4*l, 0]
    ])

    B_x = np.array([
        [0],
        [1/tau**2],
        [0],
        [-3/(4*l*tau**2)]
    ])

    # Build full 2d system (8 STATES, 2 INPUTS)

    # Zero blocks
    Z4 = np.zeros((4, 4))
    Z4x1 = np.zeros((4, 1))

    # Full A (8x8)
    A = np.block([
        [A_x, Z4],
        [Z4,  A_x]
    ])

    # Full B (8x2)
    B = np.block([
        [B_x, Z4x1],
        [Z4x1, B_x]
    ])
    
    return A, B
