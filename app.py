from flask import Flask, render_template, Response, request, jsonify
import cv2
import math
import time
import random
from ultralytics import YOLO

app = Flask(__name__)

print("[CV] Loading YOLO...")
model = YOLO("yolov8n.pt")
print("[CV] Ready.")

camera = None
camera_running = False
DISTANCE_THRESHOLD = 200
ISOLATION_TIME = 20

isolation_start_time = {}
assigned_tasks = {}

TASK_POOL = [
    "Introduce yourself to someone",
    "Share a fun fact about yourself",
    "Offer someone a drink",
    "Join the nearest activity",
    "Take a group selfie",
    "Meet someone you have not met",
    "Join the group challenge",
    "Give someone a compliment",
    "Ask for a recommendation",
    "Suggest a fun group activity",
]

# This dict is read by /stats every second
live_state = {
    "total": 0,
    "isolated": 0,
    "persons": []
}


def pick_tasks():
    return random.sample(TASK_POOL, 3)


def put_label(frame, text, x, y, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    (w, h), base = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(frame, (x - 2, y - h - 4), (x + w + 2, y + base), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, scale, text_color, thick, cv2.LINE_AA)


def generate_frames():
    global camera, camera_running, DISTANCE_THRESHOLD

    while camera_running:
        success, frame = camera.read()
        if not success:
            break

        h, w, _ = frame.shape
        results = model(frame, stream=True)

        centers = []
        boxes = []

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if cx < 40 or cx > w - 40:
                    continue
                centers.append((cx, cy))
                boxes.append((x1, y1, x2, y2, cx, cy))

        isolated_count = 0
        now = time.time()
        persons_out = []

        for i, (x1, y1, x2, y2, cx, cy) in enumerate(boxes):
            # Find nearest neighbour distance
            min_dist = float("inf")
            for j, (cx2, cy2) in enumerate(centers):
                if i == j:
                    continue
                min_dist = min(min_dist, math.hypot(cx2 - cx, cy2 - cy))

            if len(centers) == 1:
                min_dist = float("inf")

            pid = (cx // 20, cy // 20)
            label_id = f"P{i+1:02d}"
            elapsed = 0
            status = "engaged"

            if min_dist > DISTANCE_THRESHOLD:
                # Person is far from everyone
                if pid not in isolation_start_time:
                    isolation_start_time[pid] = now
                elapsed = now - isolation_start_time[pid]

                if elapsed >= ISOLATION_TIME:
                    # --- ISOLATED ---
                    status = "isolated"
                    isolated_count += 1

                    if pid not in assigned_tasks:
                        assigned_tasks[pid] = pick_tasks()
                    tasks = assigned_tasks[pid]

                    # Red box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    put_label(frame, "LOW INTERACTION",
                              x1, y1 - 12,
                              text_color=(255, 255, 255),
                              bg_color=(0, 0, 200))

                    # Tasks drawn inside box
                    ty = y1 + 20
                    for idx, task in enumerate(tasks[:3]):
                        short = task[:25] + ".." if len(task) > 25 else task
                        put_label(frame, f"{idx+1}. {short}",
                                  x1 + 4, ty,
                                  text_color=(255, 255, 160),
                                  bg_color=(40, 0, 120))
                        ty += 20

                    # Timer under box
                    put_label(frame, f"Alone: {int(elapsed)}s",
                              x1, y2 + 18,
                              text_color=(255, 80, 80),
                              bg_color=(60, 0, 0))

                else:
                    # --- WATCHING / TIMER ---
                    status = "watching"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                    put_label(frame, f"Timer: {int(elapsed)}s / {ISOLATION_TIME}s",
                              x1, y1 - 10,
                              text_color=(0, 0, 0),
                              bg_color=(0, 200, 255))

            else:
                # --- ENGAGED ---
                status = "engaged"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 2)
                put_label(frame, "Engaged",
                          x1, y1 - 10,
                          text_color=(255, 255, 255),
                          bg_color=(0, 140, 50))

                if pid in isolation_start_time:
                    del isolation_start_time[pid]
                if pid in assigned_tasks:
                    del assigned_tasks[pid]

            persons_out.append({
                "id": label_id,
                "status": status,
                "elapsed": round(elapsed, 1),
                "tasks": assigned_tasks.get(pid, [])
            })

        # HUD — top left corner
        put_label(frame, f"People: {len(boxes)}", 14, 30,
                  text_color=(255, 255, 255), bg_color=(20, 20, 20))
        put_label(frame, f"Isolated: {isolated_count}", 14, 58,
                  text_color=(80, 80, 255) if isolated_count else (80, 220, 120),
                  bg_color=(20, 20, 20))
        put_label(frame, f"Threshold: {DISTANCE_THRESHOLD}px", 14, 84,
                  text_color=(180, 180, 180), bg_color=(20, 20, 20))

        # Update shared state so /stats can serve it
        live_state["total"] = len(boxes)
        live_state["isolated"] = isolated_count
        live_state["persons"] = persons_out

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    return jsonify(live_state)


@app.route("/start_camera", methods=["POST"])
def start_camera():
    global camera, camera_running
    if not camera_running:
        camera = cv2.VideoCapture(0)
        camera_running = True
    return jsonify({"status": "started"})


@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    global camera, camera_running
    camera_running = False
    if camera:
        camera.release()
    live_state["total"] = 0
    live_state["isolated"] = 0
    live_state["persons"] = []
    return jsonify({"status": "stopped"})


@app.route("/set_distance", methods=["POST"])
def set_distance():
    global DISTANCE_THRESHOLD
    DISTANCE_THRESHOLD = int(request.json["distance"])
    return jsonify({"distance": DISTANCE_THRESHOLD})


if __name__ == "__main__":
    app.run(debug=True)
