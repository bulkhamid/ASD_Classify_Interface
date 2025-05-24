# backend/app.py

import os
import uuid
from flask import Flask, request, jsonify
from pytube import YouTube
import inference  # your inference module

# Create the Flask app, telling it that "static/" is its static folder
app = Flask(__name__, static_folder="static")

@app.route("/models", methods=["GET"])
def list_models():
    """
    List all available model names (base filenames of .pth files).
    """
    models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    try:
        model_files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
    except FileNotFoundError:
        return jsonify({"models": [], "error": "saved_models directory not found"}), 500

    model_names = [os.path.splitext(f)[0] for f in model_files]
    return jsonify({"models": model_names})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept either:
      - A file upload under the form-field "file"
      - A YouTube link under "url" or "youtube_url"
    Plus:
      - The selected model name under "model"
      - A grad-cam flag under "gradcam" ("true"/"false")

    Runs inference and returns JSON with:
      - class: predicted class label
      - label: same as class (for compatibility)
      - confidence: probability of the predicted class [0–1]
      - probabilities: dict of all class probabilities
      - video_url: URL to the annotated output video in /static/
    """
    # 1. Parse inputs
    upload = request.files.get("file")
    youtube_link = request.form.get("youtube_url") or request.form.get("url")
    model_name = request.form.get("model")
    gradcam_flag = request.form.get("gradcam", "false").lower() == "true"

    if not upload and not youtube_link:
        return jsonify({"error": "No video provided"}), 400
    if not model_name:
        return jsonify({"error": "No model specified"}), 400

    # 2. Save/download the video into static/
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)
    temp_filename = f"input_{uuid.uuid4()}.mp4"
    temp_path = os.path.join(static_dir, temp_filename)

    try:
        if upload:
            upload.save(temp_path)
        else:
            yt = YouTube(youtube_link)
            stream = yt.streams.get_highest_resolution()
            stream.download(output_path=static_dir, filename=temp_filename)
    except Exception as e:
        return jsonify({"error": f"Video retrieval failed: {e}"}), 500

    # 3. Run inference
    try:
        result = inference.run_inference(
            video_path=temp_path,
            model_name=model_name,
            gradcam=gradcam_flag
        )
    except Exception as e:
        # Clean up the temp file on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": f"Inference failed: {e}"}), 500

    # Remove the raw input video
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # 4. Build response
    label = result.get("label")
    confidence = result.get("confidence")
    # Build class-probabilities dict
    # (Assumes binary classification: ["ASD Negative", "ASD Positive"])
    prob_pos = float(confidence or 0.0)
    prob_neg = 1.0 - prob_pos
    probabilities = {
        "ASD Negative": prob_neg,
        "ASD Positive": prob_pos
    }

    response = {
        "class": label,
        "label": label,
        "confidence": prob_pos if label == "ASD Positive" else prob_neg,
        "probabilities": probabilities
    }

    # If an annotated video was produced, expose its URL
    ann_path = result.get("annotated_video")
    if ann_path:
        ann_fname = os.path.basename(ann_path)
        # Flask serves static/ at /static/
        response["video_url"] = request.host_url.rstrip("/") + f"/static/{ann_fname}"

    return jsonify(response)


if __name__ == "__main__":
    # Run the app on port 5000
    app.run(host="0.0.0.0", port=5000)
