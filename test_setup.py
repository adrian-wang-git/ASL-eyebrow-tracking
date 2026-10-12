import cv2, mediapipe
print(mediapipe.__version__)
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
print("webcam works" if ok else "webcam failed")
cap.release()