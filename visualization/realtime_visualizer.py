import cv2
import numpy as np
from perception.vision import get_measurements
from perception.camera_model import CameraModel


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