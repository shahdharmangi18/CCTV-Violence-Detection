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
ALLOWED_EXTENSIONS = {"mp4","avi","mov","mkv","webm"}

# YOUR MODEL CLASSES
VIOLENCE_CLASSES = ["violence"]
WEAPON_CLASSES = ["weapon"]

def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS


def frame_to_base64(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


def frame_to_thumbnail(frame):
    frame = cv2.resize(frame,(320,180))
    _, buffer = cv2.imencode(".jpg", frame)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


def call_roboflow(frame_b64, api_key, model_id):

    parts = model_id.split("/")
    version = parts[-1]
    model = "/".join(parts[:-1])

    url = f"https://detect.roboflow.com/{model}/{version}"
    params = {"api_key":api_key}

    r = requests.post(url, params=params, data=frame_b64)
    r.raise_for_status()

    return r.json()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status":"ok"})


@app.route("/analyze",methods=["POST"])
def analyze():

    if "video" not in request.files:
        return jsonify({"error":"No video uploaded"}),400

    video = request.files["video"]
    api_key = request.form.get("api_key")
    model_id = request.form.get("model_id")

    frame_interval = float(request.form.get("frame_interval",1))
    confidence = float(request.form.get("confidence",50)) / 100
    cluster_gap = float(request.form.get("cluster_gap",3))

    if not allowed_file(video.filename):
        return jsonify({"error":"Unsupported format"}),400

    filename = secure_filename(video.filename)
    path = os.path.join(UPLOAD_FOLDER,filename)
    video.save(path)

    cap = cv2.VideoCapture(path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    frame_step = int(fps * frame_interval)

    detections = []
    frame_index = 0
    processed = 0

    while True:

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if not ret:
            break

        timestamp = frame_index / fps

        try:

            frame_b64 = frame_to_base64(frame)
            result = call_roboflow(frame_b64, api_key, model_id)

            preds = result.get("predictions",[])

            for p in preds:

                if p["confidence"] < confidence:
                    continue

                cls = p["class"].lower()

                if cls not in ["violence","weapon"]:
                    continue

                detections.append({
                    "time":round(timestamp,2),
                    "class":cls,
                    "confidence":p["confidence"],
                    "thumbnail":frame_to_thumbnail(frame)
                })

        except:
            pass

        processed += 1
        frame_index += frame_step

        if frame_index >= total_frames:
            break

    cap.release()
    os.remove(path)

    incidents = []

    for d in detections:

        incidents.append({
            "time":d["time"],
            "end_time":d["time"],
            "type":"violence",
            "max_confidence":d["confidence"],
            "all_classes":[d["class"]],
            "thumbnail":d["thumbnail"],
            "frame_count":1,
            "frame_thumbnails":[d["thumbnail"]]
        })

    return jsonify({
        "success":True,
        "duration":round(duration,2),
        "frames_processed":processed,
        "incidents":incidents,
        "incident_count":len(incidents),
        "violence_count":len(incidents),
        "weapon_count":sum(1 for i in incidents if "weapon" in i["all_classes"])
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT",10000))
    app.run(host="0.0.0.0",port=port)