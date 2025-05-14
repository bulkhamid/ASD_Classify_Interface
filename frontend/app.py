# frontend/app.py
import streamlit as st
import requests

st.title("ASD Video Screening Demo")
st.write("Upload a child's behavior video or provide a YouTube link, select a model, and get an ASD screening prediction.")

# 1. Retrieve available models from the backend
BACKEND_URL = "http://localhost:5000"  # URL where Flask app is running
models = []
try:
    resp = requests.get(f"{BACKEND_URL}/models")
    models = resp.json().get("models", [])
except Exception as e:
    st.warning(f"Could not load model list from backend: {e}")
if models:
    model_choice = st.selectbox("Choose a model for inference:", models)
else:
    model_choice = st.text_input("Model name:", value="simple3dlstm_fold1")

# 2. Input method: upload file or YouTube link
input_mode = st.radio("Video Source:", ["Upload Video", "YouTube Link"])
video_file = None
youtube_url = None
if input_mode == "Upload Video":
    video_file = st.file_uploader("Upload a video file (.mp4, .avi, .webm)", type=["mp4","avi","webm"])
else:
    youtube_url = st.text_input("YouTube video URL:")

# 3. When the user clicks the "Run Inference" button, send request to backend
if st.button("Run Inference"):
    if not model_choice:
        st.error("Please select a model.")
    elif not video_file and not youtube_url:
        st.error("Please provide a video file or YouTube URL.")
    else:
        # Prepare request to Flask API
        files = None
        data = {"model": model_choice}
        if video_file is not None:
            # Read the file bytes and send as multipart
            files = {"video": video_file.getvalue()}
        elif youtube_url:
            data["youtube_url"] = youtube_url

        with st.spinner("Running inference..."):
            try:
                res = requests.post(f"{BACKEND_URL}/predict", files=files, data=data)
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
                st.stop()
            if res.status_code != 200:
                st.error(f"Inference failed: {res.text}")
                st.stop()
            result = res.json()
        
        # 4. Display results
        label = result.get("label", "Unknown")
        confidence = result.get("confidence", 0.0)
        st.markdown(f"**Predicted Class:** {label}  ")
        st.markdown(f"**Confidence:** {confidence*100:.1f}%")
        # Bar chart of confidence for both classes (for binary classification)
        pos_conf = confidence if "Positive" in label else 1 - confidence
        neg_conf = 1 - pos_conf
        scores = {"ASD Positive": pos_conf, "ASD Negative": neg_conf}
        st.bar_chart(scores)

        # Display annotated video if available, otherwise original video
        video_url = result.get("video_url")
        if video_url:
            st.video(video_url)
        elif video_file:
            # If no annotated video returned, at least display the uploaded video
            st.video(video_file)
