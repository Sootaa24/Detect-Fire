from ultralytics import YOLO
import cv2
import threading

class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # chỉ giữ 1 frame trong bộ đệm
        self.frame = None
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame  # luôn ghi đè frame mới nhất

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

model = YOLO('yolov8n.pt')
stream = VideoStream("rtsp://admin:24112003anh@192.168.1.64:554/Streaming/Channels/102")

while True:
    frame = stream.read()
    if frame is None:
        continue

    results = model(frame, imgsz=480, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Real-time", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()