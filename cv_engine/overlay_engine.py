import cv2
import numpy as np


class OverlayEngine:

    def __init__(self):
        pass


    def load_filter(self, path):

        filter_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if filter_img is None:
            raise Exception("Filter image not found")

        return filter_img