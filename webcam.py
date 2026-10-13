import cv2
import time

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    time_ms = int(time.time() * 1000)

    height, width, _ = frame.shape

    cv2.circle(frame, (width // 2, height // 2), 5, (0, 255, 0), -1)
    cv2.putText(frame, "NEUTRAL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)    
    cv2.putText(frame, str(time_ms), (width - 150, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)    
    
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()