import cv2
from cv_engine.face_detector import FaceDetector
from cv_engine.renderer import Renderer
def start_camera():

    cap = cv2.VideoCapture(0)

    detector = FaceDetector()
    renderer = Renderer()

    prev_x = 0
    prev_y = 0
    prev_w = 0
    prev_h = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        faces = detector.detect_faces(frame)

        for face in faces:
            x = face["x"]
            y = face["y"]
            width = face["w"]
            height = face["h"]
            confidence = face["confidence"]

            alpha = 0.7

            x = int(prev_x * alpha + x * (1-alpha))
            y = int(prev_y * alpha + y * (1-alpha))
            width = int(prev_w * alpha + width * (1-alpha))
            height = int(prev_h * alpha + height * (1-alpha))

            prev_x = x
            prev_y = y
            prev_w = width
            prev_h = height

            frame = renderer.draw_face_box(
                frame,
                x,
                y,
                width,
                height,
                confidence
            )
        
        if not faces:
            prev_x = 0
            prev_y = 0
            prev_w = 0
            prev_h = 0

        cv2.imshow("Mood Mirror", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
