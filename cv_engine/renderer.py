import cv2

class Renderer:

    def draw_face_box(self, frame, x, y, width, height, confidence):

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (255, 0, 255),
            2
        )

        confidence_text = f"{int(confidence*100)}%"

        cv2.putText(
            frame,
            confidence_text,
            (x, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2
        )

        return frame