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

    def apply_filter(self, frame, filter_img, x, y):
        h, w = filter_img.shape[:2]

        if y + h > frame.shape[0] or x + w > frame.shape[1]:
            return frame

        # If no alpha channel, create one
        if filter_img.shape[2] == 3:
            filter_rgb = filter_img
            alpha = np.ones((h, w))
        else:
            filter_rgb = filter_img[:, :, :3]
            alpha = filter_img[:, :, 3] / 255.0

        roi = frame[y:y+h, x:x+w]

        for c in range(3):
            roi[:, :, c] = (
                alpha * filter_rgb[:, :, c] +
                (1 - alpha) * roi[:, :, c]
            )

        frame[y:y+h, x:x+w] = roi

        return frame
