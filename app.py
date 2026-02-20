import cv2
import math
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Webcam
cap = cv2.VideoCapture(0)

# Settings
DISTANCE_THRESHOLD = 200
ISOLATION_TIME = 20

# Timer dictionary
isolation_start_time = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # YOLO detection
    results = model(frame, stream=True)

    centers = []
    boxes = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            # Class 0 = person
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Ignore border detections
                if cx < 40 or cx > width - 40:
                    continue

                centers.append((cx, cy))
                boxes.append((x1, y1, x2, y2, cx, cy))

    isolated_count = 0
    current_time = time.time()

    for i, (x1, y1, x2, y2, cx, cy) in enumerate(boxes):

        # Compute nearest distance
        min_distance = float("inf")
        for j, (cx2, cy2) in enumerate(centers):
            if i == j:
                continue
            distance = math.sqrt((cx2 - cx)**2 + (cy2 - cy)**2)
            min_distance = min(min_distance, distance)

        # If no neighbor → treat as isolated
        if len(centers) == 1:
            min_distance = float("inf")

        # Pseudo ID (stable-ish without tracking)
        person_id = (cx // 20, cy // 20)

        if min_distance > DISTANCE_THRESHOLD:
            # Start timer if not exists
            if person_id not in isolation_start_time:
                isolation_start_time[person_id] = current_time

            elapsed = current_time - isolation_start_time[person_id]

            if elapsed > ISOLATION_TIME:
                color = (0, 0, 255)  # RED
                isolated_count += 1
                label = "Low Interaction"
            else:
                color = (0, 255, 255)  # YELLOW waiting
                label = f"Timer: {int(elapsed)}s"
        else:
            color = (0, 255, 0)  # GREEN
            label = "Engaged"

            if person_id in isolation_start_time:
                del isolation_start_time[person_id]

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Stats
    total_people = len(boxes)
    cv2.putText(frame, f"Total People: {total_people}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"Isolated Count: {isolated_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Lonely Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()