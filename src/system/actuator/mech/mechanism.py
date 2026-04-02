import numpy as np


def wrap_angle_0_2pi(theta):
    """Wrap angle(s) in radians to [0, 2π) (hardware convention: 0°–360°)."""
    return np.mod(theta, 2 * np.pi)


class FiveBarMechanism:

    def __init__(self, transform, la, lb, min_angle_deg=0.0):
        self.tf = transform
        self.la = la
        self.lb = lb
        self.lc = transform.lc
        self.min_angle_deg = min_angle_deg
        self._min_sin = np.sin(np.radians(min_angle_deg))

    def ik(self, target_global):
        """Joint angles θ₁, θ₄ in radians, each in [0, 2π)."""

        target_l = self.tf.g2l(target_global)
        xd, yd = target_l

        la = self.la
        lb = self.lb
        lc = self.lc

        E1 = -2 * la * xd
        F1 = -2 * la * yd
        G1 = la**2 - lb**2 + xd**2 + yd**2

        disc1 = E1**2 + F1**2 - G1**2
        if disc1 < 0:
            raise ValueError("Left not reachable")

        theta1 = 2*np.arctan2((-F1 + np.sqrt(disc1)), (G1 - E1))

        E4 = 2*la*(-xd + lc)
        F4 = -2*la*yd
        G4 = lc**2 + la**2 - lb**2 + xd**2 + yd**2 - 2*lc*xd

        disc4 = E4**2 + F4**2 - G4**2
        if disc4 < 0:
            raise ValueError("Right not reachable")

        theta4 = 2*np.arctan2((-F4 - np.sqrt(disc4)), (G4 - E4))

        return wrap_angle_0_2pi(theta1), wrap_angle_0_2pi(theta4)

    def fk(self, theta1, theta4):
        """Forward kinematics; returns points in local frame."""
        la = self.la
        lb = self.lb
        lc = self.lc

        A_l = np.array([
            la*np.cos(theta1),
            la*np.sin(theta1)
        ])

        C_l = np.array([
            lc + la*np.cos(theta4),
            la*np.sin(theta4)
        ])

        d_vec = C_l - A_l
        d = np.linalg.norm(d_vec)

        if d > 2*lb:
            raise ValueError("Links cannot connect")

        e = d_vec / d
        e_perp = np.array([-e[1], e[0]])

        mid = (A_l + C_l)/2
        h = np.sqrt(lb**2 - (d/2)**2)

        P1_l = mid + h*e_perp
        P2_l = mid - h*e_perp

        return A_l, C_l, P1_l, P2_l

    def _cranks_uncrossed(self, O_l, B_l, A_l, C_l):
        """Left and right input cranks don't overlap/toggle; stay min_angle_deg away from parallel."""
        v_left_crank = A_l - O_l
        v_right_crank = C_l - B_l
        cross = np.cross(v_left_crank, v_right_crank)
        prod = np.linalg.norm(v_left_crank) * np.linalg.norm(v_right_crank)
        return cross < 0 and np.abs(cross) >= self._min_sin * prod

    def _elbows_opposed(self, O_l, B_l, A_l, C_l, P_l):
        """Left and right elbows bend in opposite directions; stay min_angle_deg away from straight."""
        la, lb = self.la, self.lb
        cross_left = np.cross(A_l - O_l, P_l - A_l)
        cross_right = np.cross(C_l - B_l, P_l - C_l)
        if cross_left * cross_right >= 0:
            return False
        return np.abs(cross_left) >= self._min_sin * la * lb and np.abs(cross_right) >= self._min_sin * la * lb

    def _coupler_above_elbows(self, A_l, C_l, P_l):
        """Coupler point P sits above the elbow-to-elbow line; stay min_angle_deg away from collinear."""
        v1 = P_l - A_l
        v2 = P_l - C_l
        cross = np.cross(v1, v2)
        prod = np.linalg.norm(v1) * np.linalg.norm(v2)
        return cross > 0 and cross >= self._min_sin * prod

    def _point_in_front(self, P_l):
        """P has positive y in the local frame (in front of the mechanism)."""
        return P_l[1] > 0

    def valid_config(self, O_l, B_l, A_l, C_l, P_l):
        return (
            self._cranks_uncrossed(O_l, B_l, A_l, C_l)
            and self._elbows_opposed(O_l, B_l, A_l, C_l, P_l)
            and self._coupler_above_elbows(A_l, C_l, P_l)
            and self._point_in_front(P_l)
        )

    def _solve_python(self, target_global):
        """Python implementation of solve(); used when Numba is unavailable or as fallback."""
        theta1, theta4 = self.ik(target_global)

        A_l, C_l, P1_l, P2_l = self.fk(theta1, theta4)

        O_l, B_l = self.tf.bases_local()

        target_l = self.tf.g2l(np.asarray(target_global))
        d1 = np.linalg.norm(target_l - P1_l)
        d2 = np.linalg.norm(target_l - P2_l)
        valid_P1 = self.valid_config(O_l, B_l, A_l, C_l, P1_l)
        valid_P2 = self.valid_config(O_l, B_l, A_l, C_l, P2_l)
        target_branch = 1 if d1 <= d2 else 2

        # Only use the branch that actually reaches the target; it must be valid (workspace/orientation).
        if target_branch == 1 and valid_P1:
            P_l = P1_l
        elif target_branch == 2 and valid_P2:
            P_l = P2_l
        else:
            raise ValueError(
                "Point not reachable in a valid configuration (outside workspace or invalid elbow orientation)"
            )

        A_g = self.tf.l2g(A_l)
        C_g = self.tf.l2g(C_l)
        P_g = self.tf.l2g(P_l)
        return theta1, theta4, A_g, C_g, P_g

    def solve(self, target_global):
        """IK + FK; θ₁, θ₄ in radians, each in [0, 2π) (0°–360°)."""
        x_g, y_g = np.asarray(target_global).flatten()[:2]
        try:
            from fivebar.numba_solve import HAS_NUMBA, get_numba_constants, solve_numba
            if HAS_NUMBA:
                if getattr(self, "_numba_constants", None) is None:
                    self._numba_constants = get_numba_constants(self)
                result = solve_numba(x_g, y_g, **self._numba_constants)
                if result[0]:
                    _, theta1, theta4, A_g_x, A_g_y, C_g_x, C_g_y, P_g_x, P_g_y = result
                    A_g = np.array([A_g_x, A_g_y])
                    C_g = np.array([C_g_x, C_g_y])
                    P_g = np.array([P_g_x, P_g_y])
                    return theta1, theta4, A_g, C_g, P_g
                raise ValueError(
                    "Point not reachable in a valid configuration (outside workspace or invalid elbow orientation)"
                )
        except (ImportError, RuntimeError):
            pass
        return self._solve_python(target_global)