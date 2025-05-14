# backend/app.py
from flask import Flask, request, jsonify, send_file
import os, uuid
import inference  # our inference module
from pytube import YouTube     # for downloading YouTube videos

app = Flask(__name__)

# (Optional) Endpoint to list available models for selection
@app.route("/models", methods=["GET"])
def list_models():
    models_dir = "backend/saved_models"
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
    # Return just base names without extension for clarity
    model_names = [os.path.splitext(f)[0] for f in model_files]
    return jsonify({"models": model_names})

@app.route("/predict", methods=["POST"])
def predict():
    # 1. Handle input: either an uploaded file or a YouTube URL
    video_file = request.files.get("video")
    youtube_url = request.form.get("youtube_url")
    if not video_file and not youtube_url:
        return jsonify({"error": "No video provided"}), 400

    # Save the video to a temporary file
    temp_dir = "backend/static"  # ensure this exists and is writable
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"input_{uuid.uuid4()}.mp4")
    try:
        if video_file:
            video_file.save(temp_path)
        else:
            # Download video from YouTube
            yt = YouTube(youtube_url)
            stream = yt.streams.get_highest_resolution()
            stream.download(output_path=os.path.dirname(temp_path), filename=os.path.basename(temp_path))
    except Exception as e:
        return jsonify({"error": f"Video retrieval failed: {e}"}), 500

    # 2. Get the selected model name from the request
    model_name = request.form.get("model")
    if model_name is None:
        # If not provided explicitly, try to infer from file name or use a default
        return jsonify({"error": "No model specified"}), 400

    # 3. Run inference (this will load the model, preprocess video, get prediction)
    try:
        result = inference.run_inference(temp_path, model_name)
    except Exception as e:
        # Cleanup and error handling
        os.remove(temp_path)
        return jsonify({"error": f"Inference failed: {e}"}), 500

    # 4. Remove the temp video file after processing
    os.remove(temp_path)

    # The result contains label, confidence, and path to annotated video
    label = result["label"]
    confidence = result["confidence"]
    ann_video_path = result.get("annotated_video")  # path to annotated video if created

    # 5. Return results. We can send the video file or provide a URL reference.
    response_data = {
        "label": label,
        "confidence": confidence
    }
    if ann_video_path:
        # Option 1: Return a direct URL to the video (assuming static folder is exposed)
        response_data["video_url"] = request.host_url + "static/" + os.path.basename(ann_video_path)
        # Option 2: Send the video file directly (commented out):
        # return send_file(ann_video_path, mimetype="video/mp4")
    return jsonify(response_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
# Note: Ensure that the backend/saved_models directory exists and contains the model files.     
# Also, ensure that the backend/static directory exists and is writable for saving temporary files.