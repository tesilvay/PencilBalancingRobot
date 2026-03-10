


class CameraModel:
    """
    Converts between normalized line parameters (b,s)
    and pixel line parameters (a_px, b_px).
    """

    def __init__(self, width=346, height=260, fx=None, fy=None, cx=None, cy=None):

        self.width = width
        self.height = height

        self.fx = fx if fx is not None else width / 2
        self.fy = fy if fy is not None else height / 2

        self.cx = cx if cx is not None else width / 2
        self.cy = cy if cy is not None else height / 2

    def normalized_to_pixel(self, b, s):

        s_px = (self.fx / self.fy) * s
        a_px = self.fx * b - (self.fx / self.fy) * s * self.cy + self.cx

        return a_px, s_px

    def pixel_to_normalized(self, a_px, b_px):

        s = b_px * self.fy / self.fx
        b = (a_px - self.cx + b_px * self.cy) / self.fx

        return b, s