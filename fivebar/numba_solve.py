"""
Numba JIT-compiled kernel for five-bar reachability + IK + FK + valid_config.
Used by workspace sweeps to reduce per-point cost. Falls back to Python mech.solve when Numba is unavailable.
"""
import math
import numpy as np

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:

    @numba.njit(fastmath=True)
    def point_valid_numba(x_g, y_g, o_g_0, o_g_1, rt_00, rt_01, rt_10, rt_11, lc, la, lb, r_min, r_max, min_sin):
        """
        Returns True iff the global point (x_g, y_g) is reachable in a valid configuration.
        Constants: O_g = (o_g_0, o_g_1), R_T 2x2 = (rt_00, rt_01; rt_10, rt_11), lc, la, lb, r_min, r_max, min_sin.
        """
        # g2l: target_l = R_T @ (p_global - O_g)
        dx = x_g - o_g_0
        dy = y_g - o_g_1
        xd = rt_00 * dx + rt_01 * dy
        yd = rt_10 * dx + rt_11 * dy

        # Reachability (distance from O_l and B_l)
        r_O = math.sqrt(xd * xd + yd * yd)
        r_B = math.sqrt((xd - lc) * (xd - lc) + yd * yd)
        if r_O < r_min or r_O > r_max or r_B < r_min or r_B > r_max:
            return False

        # IK left
        E1 = -2.0 * la * xd
        F1 = -2.0 * la * yd
        G1 = la * la - lb * lb + xd * xd + yd * yd
        disc1 = E1 * E1 + F1 * F1 - G1 * G1
        if disc1 < 0.0:
            return False
        theta1 = 2.0 * math.atan2(-F1 + math.sqrt(disc1), G1 - E1)

        # IK right
        E4 = 2.0 * la * (-xd + lc)
        F4 = -2.0 * la * yd
        G4 = lc * lc + la * la - lb * lb + xd * xd + yd * yd - 2.0 * lc * xd
        disc4 = E4 * E4 + F4 * F4 - G4 * G4
        if disc4 < 0.0:
            return False
        theta4 = 2.0 * math.atan2(-F4 - math.sqrt(disc4), G4 - E4)

        # FK
        ax = la * math.cos(theta1)
        ay = la * math.sin(theta1)
        cx = lc + la * math.cos(theta4)
        cy = la * math.sin(theta4)
        d_vec_x = cx - ax
        d_vec_y = cy - ay
        d = math.sqrt(d_vec_x * d_vec_x + d_vec_y * d_vec_y)
        if d > 2.0 * lb:
            return False
        e_x = d_vec_x / d
        e_y = d_vec_y / d
        e_perp_x = -e_y
        e_perp_y = e_x
        mid_x = 0.5 * (ax + cx)
        mid_y = 0.5 * (ay + cy)
        h = math.sqrt(lb * lb - (d * 0.5) ** 2)
        p1_x = mid_x + h * e_perp_x
        p1_y = mid_y + h * e_perp_y
        p2_x = mid_x - h * e_perp_x
        p2_y = mid_y - h * e_perp_y

        # Branch choice by distance to target_l (xd, yd)
        d1_sq = (xd - p1_x) ** 2 + (yd - p1_y) ** 2
        d2_sq = (xd - p2_x) ** 2 + (yd - p2_y) ** 2
        O_l_x, O_l_y = 0.0, 0.0
        B_l_x, B_l_y = lc, 0.0

        if d1_sq <= d2_sq:
            px, py = p1_x, p1_y
        else:
            px, py = p2_x, p2_y

        # valid_config(O_l, B_l, A_l, C_l, P_l) — 2D cross = ax*by - ay*bx
        # _cranks_uncrossed
        v_left_x = ax - O_l_x
        v_left_y = ay - O_l_y
        v_right_x = cx - B_l_x
        v_right_y = cy - B_l_y
        cross_c = v_left_x * v_right_y - v_left_y * v_right_x
        n_left = math.sqrt(v_left_x * v_left_x + v_left_y * v_left_y)
        n_right = math.sqrt(v_right_x * v_right_x + v_right_y * v_right_y)
        prod = n_left * n_right
        if cross_c >= 0.0 or math.fabs(cross_c) < min_sin * prod:
            return False

        # _elbows_opposed
        cross_left = (ax - O_l_x) * (py - ay) - (ay - O_l_y) * (px - ax)
        cross_right = (cx - B_l_x) * (py - cy) - (cy - B_l_y) * (px - cx)
        if cross_left * cross_right >= 0.0:
            return False
        if math.fabs(cross_left) < min_sin * la * lb or math.fabs(cross_right) < min_sin * la * lb:
            return False

        # _coupler_above_elbows
        v1_x = px - ax
        v1_y = py - ay
        v2_x = px - cx
        v2_y = py - cy
        cross_cp = v1_x * v2_y - v1_y * v2_x
        n1 = math.sqrt(v1_x * v1_x + v1_y * v1_y)
        n2 = math.sqrt(v2_x * v2_x + v2_y * v2_y)
        if cross_cp <= 0.0 or cross_cp < min_sin * n1 * n2:
            return False

        # _point_in_front
        if py <= 0.0:
            return False

        return True

else:
    # Stub when Numba not installed; callers should use Python path.
    def point_valid_numba(x_g, y_g, o_g_0, o_g_1, rt_00, rt_01, rt_10, rt_11, lc, la, lb, r_min, r_max, min_sin):
        raise RuntimeError("Numba is not installed; use Python sweep path.")


def get_numba_constants(mech):
    """
    Build flat constants for point_valid_numba from a FiveBarMechanism.
    Returns a dict with keys: o_g_0, o_g_1, rt_00, rt_01, rt_10, rt_11, lc, la, lb, r_min, r_max, min_sin.
    """
    tf = mech.tf
    O_g = tf.O_g
    R_T = tf.R.T
    lc = float(tf.lc)
    la = float(mech.la)
    lb = float(mech.lb)
    r_min = max(0.0, abs(la - lb) * 0.95)
    r_max = (la + lb) * 1.05
    min_sin = float(mech._min_sin)
    return {
        "o_g_0": float(O_g[0]),
        "o_g_1": float(O_g[1]),
        "rt_00": float(R_T[0, 0]),
        "rt_01": float(R_T[0, 1]),
        "rt_10": float(R_T[1, 0]),
        "rt_11": float(R_T[1, 1]),
        "lc": lc,
        "la": la,
        "lb": lb,
        "r_min": r_min,
        "r_max": r_max,
        "min_sin": min_sin,
    }
