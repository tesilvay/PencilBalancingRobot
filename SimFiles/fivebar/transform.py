import numpy as np


class FiveBarTransform:
    """
    Handles coordinate transforms between local (paper) frame
    and global (mechanism) frame.
    """

    def __init__(self, O_global, B_global):

        self.O_g = np.array(O_global, dtype=float)
        self.B_g = np.array(B_global, dtype=float)

        OB = self.B_g - self.O_g
        self.lc = np.linalg.norm(OB)

        self.phi = np.arctan2(OB[1], OB[0])

        c = np.cos(self.phi)
        s = np.sin(self.phi)

        self.R = np.array([
            [c, -s],
            [s,  c]
        ])

        self.O_l = np.array([0.0, 0.0])
        self.B_l = np.array([self.lc, 0.0])

    def l2g(self, p_local):
        p_local = np.array(p_local)
        return self.R @ p_local + self.O_g

    def g2l(self, p_global):
        p_global = np.array(p_global)
        return self.R.T @ (p_global - self.O_g)

    def bases_local(self):
        return self.O_l, self.B_l

    def bases_global(self):
        return self.O_g, self.B_g