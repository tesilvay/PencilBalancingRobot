


from core.sim_types import CameraObservation


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

    def normalized_to_pixel(self, obs: CameraObservation):
        
        s = obs.slope
        b = obs.intercept

        s_px = (self.fx / self.fy) * s
        b_px = self.fx * b - (self.fx / self.fy) * s * self.cy + self.cx

        return CameraObservation(
            slope=s_px,
            intercept=b_px
        )

    def pixel_to_normalized(self, obs_px: CameraObservation):

        s = obs_px.slope
        b = obs_px.intercept

        s = s * self.fy / self.fx

        b = (b - self.cx) / self.fx

        return CameraObservation(
            slope=s,
            intercept=b
        )