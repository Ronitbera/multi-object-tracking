import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize tracker
tracker = DeepSort(max_age=30)

# Load video
cap = cv2.VideoCapture("input.mp4")

# Output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, 30, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    detections = []

    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r

        if int(class_id) == 0:  # person
            detections.append(([x1, y1, x2-x1, y2-y1], score, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = map(int, track.to_ltrb())

        cv2.rectangle(frame, (l, t), (l+w, t+h), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (l, t-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    frame = cv2.resize(frame, (640, 480))
    out.write(frame)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()