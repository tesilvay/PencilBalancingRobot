import numpy as np
from core.sim_types import PhysicalParams
import control as ct

def BuildLinearModel(param: PhysicalParams):
    g = param.g
    l = param.com_length
    tau = param.tau
    zeta = param.zeta
    
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
