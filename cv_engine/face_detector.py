import cv2
import mediapipe as mp

class FaceDetector:

    def __init__(self):

        self.mp_face = mp.solutions.face_detection

        self.detector = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )

    def detect_faces(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.detector.process(rgb_frame)

        faces = []

        if results and results.detections:

            h, w, c = frame.shape

            for detection in results.detections:

                bbox = detection.location_data.relative_bounding_box

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                confidence = detection.score[0]

                faces.append({
                    "x": x,
                    "y": y,
                    "w": width,
                    "h": height,
                    "confidence": confidence
                })

        return faces