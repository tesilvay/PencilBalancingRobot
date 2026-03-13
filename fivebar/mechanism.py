import numpy as np


class FiveBarMechanism:

    def __init__(self, transform, la, lb):

        self.tf = transform
        self.la = la
        self.lb = lb
        self.lc = transform.lc

    def ik(self, target_global):

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

        return theta1, theta4

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
        """Left and right input cranks don't overlap/toggle."""
        v_left_crank = A_l - O_l
        v_right_crank = C_l - B_l
        return np.cross(v_left_crank, v_right_crank) < 0

    def _elbows_opposed(self, O_l, B_l, A_l, C_l, P_l):
        """Left and right elbows bend in opposite directions."""
        cross_left = np.cross(A_l - O_l, P_l - A_l)
        cross_right = np.cross(C_l - B_l, P_l - C_l)
        return cross_left * cross_right < 0

    def _coupler_above_elbows(self, A_l, C_l, P_l):
        """Coupler point P sits above the elbow-to-elbow line."""
        return np.cross(P_l - A_l, P_l - C_l) > 0

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

    def solve(self, target_global):

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