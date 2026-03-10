import cv2
import numpy as np
from perception.vision import get_measurements


class CameraModel:
    """
    Converts between normalized line parameters (b,s)
    and pixel line parameters (a_px, b_px).
    """

    def __init__(self, width, height, fx=None, fy=None, cx=None, cy=None):

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


class PencilVisualizerRealtime:

    def __init__(self, width=346, height=260):

        self.width = width
        self.height = height

        self.cam = CameraModel(width, height)
        
        self.cam_x = "Camera x-axis"
        self.cam_y = "Camera y-axis"
        

        cv2.namedWindow(self.cam_x, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.cam_y, cv2.WINDOW_NORMAL)

        cv2.moveWindow(self.cam_x, 50, 100)
        cv2.moveWindow(self.cam_y, 50 + self.width + 55, 137)

    def draw_line(self, img, b, s):

        # Convert normalized → pixel
        a_px, b_px = self.cam.normalized_to_pixel(b, s)

        y0 = 0
        y1 = self.height - 1

        x0 = int(a_px + b_px * y0)
        x1 = int(a_px + b_px * y1)

        cv2.line(img, (x0, y0), (x1, y1), 255, 2)

    def render(self, measurement):

        if measurement is None:
            return

        img1 = np.zeros((self.height, self.width), dtype=np.uint8)
        img2 = np.zeros((self.height, self.width), dtype=np.uint8)

        b1, s1, b2, s2 = get_measurements(measurement)

        self.draw_line(img1, b1, s1)
        self.draw_line(img2, b2, s2)

        cv2.imshow(self.cam_x, img1)
        cv2.imshow(self.cam_y, img2)

        cv2.waitKey(1)