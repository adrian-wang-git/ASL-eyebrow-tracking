import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)

landmarker = vision.FaceLandmarker.create_from_options(options)

BROW_POINTS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
BROW_SCORES = ["browInnerUp", "browOuterUpLeft", "browOuterUpRight",
               "browDownLeft", "browDownRight"]
RAISED_BROW_THRESHOLD = 0.20
LOWERED_BROW_THRESHOLD = 0.15

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    time_ms = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, time_ms)

    height, width, _ = frame.shape

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        for i in BROW_POINTS:
            landmark = landmarks[i]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        scores = {c.category_name: c.score for c in result.face_blendshapes[0]}

        text_y_cord = 30
        for name in BROW_SCORES:
            cv2.putText(frame, f"{name}: {scores[name]:.2f}", (10, text_y_cord),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            text_y_cord += 25

        raised_avg_score = (scores["browInnerUp"] + scores["browOuterUpLeft"] + scores["browOuterUpRight"]) / 3
        lowered_avg_score = (scores["browDownLeft"] + scores["browDownRight"]) / 2

        if raised_avg_score > RAISED_BROW_THRESHOLD:
            cv2.putText(frame, "RAISED", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
        elif lowered_avg_score > LOWERED_BROW_THRESHOLD:
            cv2.putText(frame, "LOWERED", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "NEUTRAL", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)

    else:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.putText(frame, str(time_ms), (width - 150, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)    
    
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()