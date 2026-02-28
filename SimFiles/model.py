import numpy as np
from sim_types import PhysicalParams

def BuildLinearModel(param: PhysicalParams):
    g = param.g
    l = param.com_length
    tau = param.tau
    zeta = param.zeta
    
    # -----------------------------
    # STATE VECTOR DEFINITION (ONE AXIS)
    # -----------------------------

    """
    We define the state vector for ONE AXIS (x-direction) as:

        x1 = x              (table position)
        x2 = x_dot          (table velocity)
        x3 = alpha          (pencil angle)
        x4 = alpha_dot      (pencil angular velocity)

    So the state vector is:

        X = [ x, x_dot, alpha, alpha_dot ]^T

    The input is:

        u = x_des

    Goal:
    Convert second-order equations into first-order form:

        X_dot = A_x * X + B_x * u
    """


    # -----------------------------
    # ORIGINAL SECOND-ORDER EQUATIONS
    # -----------------------------

    """
    Table dynamics:

        tau^2 * x_ddot + 2*zeta*tau * x_dot + x = u

    Rewritten:

        x_ddot = (1/tau^2)*(u - x) - (2*zeta/tau)*x_dot

    Pencil dynamics:

        alpha_ddot = (g/l)*alpha - (1/l)*x_ddot

    Substitute x_ddot into pencil equation:

        alpha_ddot =
            (g/l)*alpha
            - (1/l)*[(1/tau^2)*(u - x) - (2*zeta/tau)*x_dot]
    """


    # -----------------------------
    # CONVERT TO FIRST-ORDER FORM
    # -----------------------------

    """
    First-order equations:

    1) x1_dot = x2

    2) x2_dot =
        (1/tau^2)*(u - x1)
        - (2*zeta/tau)*x2

    3) x3_dot = x4

    4) x4_dot =
        (g/l)*x3
        - (1/l)*[(1/tau^2)*(u - x1)
                    - (2*zeta/tau)*x2]

    Now we collect terms in form:

        X_dot = A_x * X + B_x * u
    """


    # -----------------------------
    # BUILD A_x MATRIX (4x4, ONE AXIS)
    # -----------------------------

    A_x = np.array([
        [0, 1, 0, 0],
        [-1/tau**2, -2*zeta/tau, 0, 0],
        [0, 0, 0, 1],
        [1/(l*tau**2), 2*zeta/(l*tau), g/l, 0]
    ])


    # -----------------------------
    # BUILD B_x VECTOR (4x1, ONE AXIS)
    # -----------------------------

    B_x = np.array([
        [0],
        [1/tau**2],
        [0],
        [-1/(l*tau**2)]
    ])


    # -----------------------------
    # BUILD FULL 2D SYSTEM (8 STATES, 2 INPUTS)
    # -----------------------------

    """
    Since x and y directions are decoupled in our simplified model,
    the total A matrix is block diagonal:

            [ A_x   0  ]
        A = [  0   A_y ]

    Where A_y = A_x (same dynamics)

    Similarly, B becomes:

            [ B_x   0  ]
        B = [  0   B_y ]

    Inputs:
        u = [x_des, y_des]^T
    """

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
