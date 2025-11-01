from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture("rtsp://admin:24112003anh@192.168.1.64:554/Streaming/Channels/102")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    for i in range(2):
        cap.grab()
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()