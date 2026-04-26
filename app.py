import os
import cv2
import base64
import requests
import tempfile
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder="static")
CORS(app)

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def frame_to_base64(frame):
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode("utf-8")

def frame_to_thumbnail(frame, width=320):
    h, w = frame.shape[:2]
    ratio = width / w
    new_h = int(h * ratio)
    resized = cv2.resize(frame, (width, new_h))
    _, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

def call_roboflow(frame_b64, api_key, model_id, confidence):
    parts = model_id.strip("/").split("/")
    if len(parts) < 2:
        model_name = parts[0]
        version = "1"
    else:
        version = parts[-1]
        model_name = "/".join(parts[:-1])

    url = f"https://detect.roboflow.com/{model_name}/{version}"
    params = {"api_key": api_key, "confidence": int(confidence)}

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(url, params=params, data=frame_b64, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()

def cluster_detections(detections, gap_secs=3.0):
    if not detections:
        return []
    clusters = []
    cur = {**detections[0], "frames": [detections[0]]}
    for d in detections[1:]:
        if d["time"] - cur["time"] <= gap_secs:
            cur["frames"].append(d)
            cur["end_time"] = d["time"]
            if len(d["predictions"]) > len(cur["predictions"]):
                cur["predictions"] = d["predictions"]
                cur["thumbnail"] = d["thumbnail"]
        else:
            clusters.append(cur)
            cur = {**d, "frames": [d]}
    clusters.append(cur)

    result = []
    for c in clusters:
        all_classes = list({p["class"] for f in c["frames"] for p in f["predictions"]})
        max_conf = max((p["confidence"] for f in c["frames"] for p in f["predictions"]), default=0)
        is_violence = any(
            kw in cl.lower()
            for cl in all_classes
            for kw in ["violence", "fight", "aggress", "assault", "attack"]
        )
        if is_violence:
          inc_type = "violence"
        elif max_conf > 0.6:
          inc_type = "harassment"
        else:
          inc_type = "normal"
        result.append({
            "time": c["time"],
            "end_time": c.get("end_time", c["time"]),
            "type": inc_type,
            "predictions": c["predictions"],
            "all_classes": all_classes,
            "max_confidence": round(max_conf, 3),
            "frame_count": len(c["frames"]),
            "thumbnail": c.get("thumbnail", ""),
            "frame_thumbnails": [f.get("thumbnail", "") for f in c["frames"][:6]],
        })
    return result

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    api_key = request.form.get("api_key", "").strip()
    model_id = request.form.get("model_id", "").strip()
    frame_interval = float(request.form.get("frame_interval", 1.0))
    confidence = float(request.form.get("confidence", 50))
    cluster_gap = float(request.form.get("cluster_gap", 3.0))

    if not api_key:
        return jsonify({"error": "Roboflow API key is required"}), 400
    if not model_id:
        return jsonify({"error": "Model ID is required"}), 400
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid or unsupported video file"}), 400

    filename = secure_filename(file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)

    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        frame_step = max(1, int(fps * frame_interval))

        detections = []
        frame_idx = 0
        processed = 0
        errors = 0

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            frame_b64 = frame_to_base64(frame)

            try:
                result = call_roboflow(frame_b64, api_key, model_id, confidence)
                preds = result.get("predictions", [])
                if preds:
                  violence_counter += 1
                else:
                 violence_counter = 0

# Only trigger detection if violence appears in 3 frames
                if violence_counter >= 3:
                  thumb = frame_to_thumbnail(frame)
                  detections.append({
                  "time": round(timestamp, 2),
                  "predictions": preds,
                   "thumbnail": thumb,
                 })
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code in (401, 403):
                    cap.release()
                    os.remove(video_path)
                    return jsonify({"error": "Invalid API key or model not found. Check your Roboflow credentials."}), 401
                errors += 1
            except Exception:
                errors += 1

            processed += 1
            frame_idx += frame_step
            if frame_idx >= total_frames:
                break

        cap.release()
        os.remove(video_path)

        incidents = cluster_detections(detections, cluster_gap)

        return jsonify({
            "success": True,
            "duration": round(duration, 2),
            "frames_processed": processed,
            "total_frames": total_frames,
            "incidents": incidents,
            "incident_count": len(incidents),
            "violence_count": sum(1 for i in incidents if i["type"] == "violence"),
            "harassment_count": sum(1 for i in incidents if i["type"] == "harassment"),
            "errors": errors,
        })

    except Exception as e:
        if os.path.exists(video_path):
            os.remove(video_path)
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "sentinel-backend"})


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(host="0.0.0.0", port=10000)
