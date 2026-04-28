import os
import cv2
import base64
import requests
import tempfile
import subprocess
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {"mp4","avi","mov","mkv","webm"}

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
    params = {"api_key": api_key}

    r = requests.post(url, params=params, data=frame_b64)
    r.raise_for_status()

    return r.json()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze():

    if "video" not in request.files:
        return jsonify({"error":"No video uploaded"}),400

    video = request.files["video"]
    api_key = request.form.get("api_key")
    model_id = request.form.get("model_id")

    frame_interval = float(request.form.get("frame_interval",0.5))
    confidence = float(request.form.get("confidence",10)) / 100

    if not allowed_file(video.filename):
        return jsonify({"error":"Unsupported format"}),400

    filename = secure_filename(video.filename)
    video_path = os.path.join(UPLOAD_FOLDER,filename)
    video.save(video_path)

    frames_dir = os.path.join(UPLOAD_FOLDER,"frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Extract frames using ffmpeg
    subprocess.run([
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps=1/{frame_interval}",
        os.path.join(frames_dir,"frame_%04d.jpg")
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frame_files = sorted(os.listdir(frames_dir))

    incidents = []
    processed = 0

    for f in frame_files:

        frame_path = os.path.join(frames_dir,f)
        frame = cv2.imread(frame_path)

        if frame is None:
            continue

        processed += 1

        try:

            frame_b64 = frame_to_base64(frame)
            result = call_roboflow(frame_b64, api_key, model_id)

            preds = result.get("predictions",[])

            print(result)
            for p in preds:

             cls = p["class"].lower()

              # accept lower confidence detections
             if p["confidence"] < 0.08:
              continue

            cls = p["class"].lower()

            if "violence" in cls or "weapon" in cls:

                 incidents.append({
                    "time": processed,
                    "type": cls,
                    "max_confidence": p["confidence"],
                    "all_classes":[cls],
                    "thumbnail": frame_to_thumbnail(frame)
                })

        except Exception as e:
            print("Detection error:",e)

    # cleanup
    try:
        os.remove(video_path)
    except:
        pass

    for f in frame_files:
        try:
            os.remove(os.path.join(frames_dir,f))
        except:
            pass

    return jsonify({
        "success": True,
        "frames_processed": processed,
        "incidents": incidents,
        "incident_count": len(incidents),
        "violence_count": sum(1 for i in incidents if "violence" in i["all_classes"]),
        "weapon_count": sum(1 for i in incidents if "weapon" in i["all_classes"])
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT",10000))
    app.run(host="0.0.0.0", port=port)