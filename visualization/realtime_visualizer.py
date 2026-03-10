import cv2
import numpy as np
from perception.vision import get_measurements


class PencilVisualizerRealtime:

    def __init__(self, width=346, height=260):

        self.width = width
        self.height = height

        cv2.namedWindow("Camera 1", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Camera 2", cv2.WINDOW_NORMAL)

    def draw_line(self, img, b, s):

        """
        Draw line x = s*y + b
        """

        h, w = img.shape

        y1 = 0
        y2 = h - 1

        x1 = int(s * y1 + b)
        x2 = int(s * y2 + b)

        cv2.line(img, (x1, y1), (x2, y2), 255, 2)

    def render(self, measurement):

        if measurement is None:
            return

        img1 = np.zeros((self.height, self.width), dtype=np.uint8)
        img2 = np.zeros((self.height, self.width), dtype=np.uint8)

        b1, s1, b2, s2 = get_measurements(measurement)

        self.draw_line(img1, b1, s1)
        self.draw_line(img2, b2, s2)

        cv2.imshow("Camera 1", img1)
        cv2.imshow("Camera 2", img2)

        cv2.waitKey(1)