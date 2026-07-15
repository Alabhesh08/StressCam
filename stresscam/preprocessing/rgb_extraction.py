import cv2
import numpy as np


class RGBExtractor:

    def extract(self, frame, mask):
        """
        Returns mean BGR and RGB values over masked pixels.
        """

        pixels = frame[mask > 0]

        if len(pixels) == 0:
            return None

        mean_bgr = pixels.mean(axis=0)

        mean_rgb = mean_bgr[::-1]

        return mean_rgb