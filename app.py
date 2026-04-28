import os
import cv2
import base64
import requests
import tempfile
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

# ─────────────────────────────────────────────────────────────────
# IMPORTANT: These are the class names your Roboflow model returns.
# Use /debug-frame endpoint to see your model's EXACT class names,
# then edit these lists to match. Case-insensitive matching is used.
# ─────────────────────────────────────────────────────────────────
VIOLENCE_CLASSES = [
    "violence", "violent", "fight", "fighting",
    "attack", "attacking", "aggression", "aggressive",
]

WEAPON_CLASSES = [
    "weapon", "gun", "knife", "pistol",
    "rifle", "sword", "blade",
]

ALL_ALERT_CLASSES = set(VIOLENCE_CLASSES + WEAPON_CLASSES)


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
        model_name, version = parts[0], "1"
    else:
        version = parts[-1]
        model_name = "/".join(parts[:-1])

    url = f"https://detect.roboflow.com/{model_name}/{version}"
    params = {"api_key": api_key, "confidence": int(confidence)}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    resp = requests.post(url, params=params, data=frame_b64, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


def is_alert_class(class_name: str) -> bool:
    """Return True only if this class is in our known alert list."""
    return class_name.strip().lower() in ALL_ALERT_CLASSES


def classify_incident_type(all_classes):
    for cl in all_classes:
        if cl.strip().lower() in set(WEAPON_CLASSES):
            return "weapon"
    for cl in all_classes:
        if cl.strip().lower() in set(VIOLENCE_CLASSES):
            return "violence"
    return None


def cluster_detections(detections, gap_secs=3.0):
    if not detections:
        return []

    clusters = []
    cur = {**detections[0], "frames": [detections[0]]}

    for d in detections[1:]:
        if d["time"] - cur["time"] <= gap_secs:
            cur["frames"].append(d)
            cur["end_time"] = d["time"]
            if d.get("max_conf", 0) > cur.get("max_conf", 0):
                cur["thumbnail"] = d["thumbnail"]
                cur["max_conf"] = d["max_conf"]
        else:
            clusters.append(cur)
            cur = {**d, "frames": [d]}
    clusters.append(cur)

    result = []
    for c in clusters:
        all_classes = list({
            p["class"] for f in c["frames"]
            for p in f["predictions"]
            if is_alert_class(p["class"])
        })

        # Skip clusters with no actual alert classes
        if not all_classes:
            continue

        max_conf = max(
            (p["confidence"] for f in c["frames"] for p in f["predictions"]
             if is_alert_class(p["class"])),
            default=0
        )

        inc_type = classify_incident_type(all_classes)
        if not inc_type:
            continue

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


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    api_key = request.form.get("api_key", "").strip()
    model_id = request.form.get("model_id", "").strip()
    frame_interval = float(request.form.get("frame_interval", 1))
    confidence = float(request.form.get("confidence", 50))
    cluster_gap = float(request.form.get("cluster_gap", 3))

    if not api_key:
        return jsonify({"error": "API key is required"}), 400
    if not model_id:
        return jsonify({"error": "Model ID is required"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file format"}), 400

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

                # FIX 1: Filter predictions below confidence threshold
                conf_threshold = confidence / 100.0
                preds = [p for p in preds if p.get("confidence", 0) >= conf_threshold]

                # FIX 2: Only keep predictions whose class is in our alert list
                alert_preds = [p for p in preds if is_alert_class(p["class"])]

                if alert_preds:
                    thumb = frame_to_thumbnail(frame)
                    max_conf = max(p["confidence"] for p in alert_preds)
                    detections.append({
                        "time": round(timestamp, 2),
                        "predictions": alert_preds,
                        "thumbnail": thumb,
                        "max_conf": max_conf,
                    })

            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code in (401, 403):
                    cap.release()
                    os.remove(video_path)
                    return jsonify({"error": "Invalid API key or model not accessible."}), 401
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
            "weapon_count": sum(1 for i in incidents if i["type"] == "weapon"),
            "errors": errors,
        })

    except Exception as e:
        if os.path.exists(video_path):
            os.remove(video_path)
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────
# DEBUG ENDPOINT — call this first to see what class names your
# model actually returns. Then update VIOLENCE_CLASSES above.
#
# curl -X POST https://your-site.onrender.com/debug-frame \
#   -F "image=@frame.jpg" \
#   -F "api_key=rf_xxx" \
#   -F "model_id=workspace/model/1"
# ─────────────────────────────────────────────────────────────────
@app.route("/debug-frame", methods=["POST"])
def debug_frame():
    api_key = request.form.get("api_key", "").strip()
    model_id = request.form.get("model_id", "").strip()

    if not api_key or not model_id:
        return jsonify({"error": "api_key and model_id are required"}), 400

    img_file = request.files.get("image")
    if not img_file:
        return jsonify({"error": "Provide an image file with field name 'image'"}), 400

    frame_b64 = base64.b64encode(img_file.read()).decode("utf-8")

    try:
        result = call_roboflow(frame_b64, api_key, model_id, confidence=1)
        preds = result.get("predictions", [])
        unique_classes = list({p["class"] for p in preds})

        return jsonify({
            "raw_response": result,
            "all_classes_returned": unique_classes,
            "alert_classes_matched": [c for c in unique_classes if is_alert_class(c)],
            "non_alert_classes": [c for c in unique_classes if not is_alert_class(c)],
            "tip": (
                "If 'alert_classes_matched' is empty but 'non_alert_classes' has entries, "
                "copy those exact class names into VIOLENCE_CLASSES or WEAPON_CLASSES in app.py"
            ),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "violence_classes_configured": VIOLENCE_CLASSES,
        "weapon_classes_configured": WEAPON_CLASSES,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
