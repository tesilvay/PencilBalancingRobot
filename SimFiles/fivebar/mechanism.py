import numpy as np


class FiveBarMechanism:

    def __init__(self, transform, la, lb):

        self.tf = transform
        self.la = la
        self.lb = lb
        self.lc = transform.lc

    def ik(self, target_global):

        p = self.tf.g2l(target_global)
        xd, yd = p

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

        la = self.la
        lb = self.lb
        lc = self.lc

        A = np.array([
            la*np.cos(theta1),
            la*np.sin(theta1)
        ])

        C = np.array([
            lc + la*np.cos(theta4),
            la*np.sin(theta4)
        ])

        d_vec = C - A
        d = np.linalg.norm(d_vec)

        if d > 2*lb:
            raise ValueError("Links cannot connect")

        e = d_vec / d
        e_perp = np.array([-e[1], e[0]])

        mid = (A + C)/2
        h = np.sqrt(lb**2 - (d/2)**2)

        P1 = mid + h*e_perp
        P2 = mid - h*e_perp

        return A, C, P1, P2

    def valid_config(self, O_l, B_l, A_l, C_l, P_l):

        v1 = A_l - O_l
        v2 = P_l - A_l
        v3 = C_l - B_l
        v4 = P_l - C_l

        cross = np.cross(v1, v3)
        cond1 = cross < 0

        cross_left = np.cross(v1, v2)
        cross_right = np.cross(v3, v4)
        cond2 = cross_left * cross_right < 0

        cross_in = np.cross(v2, v4)
        cond3 = cross_in > 0

        return cond1 and cond2 and cond3

    def solve(self, target_global):

        theta1, theta4 = self.ik(target_global)

        A, C, P1, P2 = self.fk(theta1, theta4)

        O_l, B_l = self.tf.bases_local()

        if self.valid_config(O_l, B_l, A, C, P1):
            P = P1
        else:
            P = P2

        return theta1, theta4, A, C, P