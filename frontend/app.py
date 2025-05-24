# frontend/app.py
import streamlit as st
import requests
import pandas as pd

# Configure page
st.set_page_config(page_title="Video Inference Demo", layout="centered")

# Backend API endpoint (update if needed)
BACKEND_URL = "http://localhost:5000"  # Example backend URL

st.title("Video Inference with Grad-CAM")

# Sidebar or top controls for Grad-CAM option
gradcam_enabled = st.checkbox("Enable Grad-CAM visualization", value=False)

# Mode selection
mode = st.radio("Choose input mode:", ["Upload Video", "YouTube Link", "YouTube + Webcam"], index=0)

# Utility function to extract YouTube video ID from a URL
def extract_youtube_id(url: str) -> str:
    """Extract the YouTube video ID from a YouTube URL."""
    if url is None:
        return None
    url = url.strip()
    if url == "":
        return None
    # Support youtu.be short links and full links
    if "youtu.be/" in url:
        # short link format
        vid_id = url.split("/")[-1].split("?")[0]
        return vid_id
    if "watch?v=" in url:
        # full link format
        start = url.find("watch?v=") + len("watch?v=")
        end = url.find("&", start)
        vid_id = url[start:] if end == -1 else url[start:end]
        return vid_id
    if "/embed/" in url:
        # already an embed link
        start = url.find("/embed/") + len("/embed/")
        end = url.find("?", start)
        vid_id = url[start:] if end == -1 else url[start:end]
        return vid_id
    # If the URL is not recognized as YouTube, return None
    return None

# Ensure session state for recording state
if "start_record" not in st.session_state:
    st.session_state["start_record"] = False

# If user switches mode away from webcam, reset the recording state
if mode != "YouTube + Webcam" and st.session_state.get("start_record"):
    st.session_state["start_record"] = False

# Mode-specific UI and logic
if mode == "Upload Video":
    # File uploader for video files
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv", "webm"])
    if uploaded_file is not None:
        # Once a file is uploaded, enable running inference
        if st.button("Run Inference"):
            # Read file bytes
            file_bytes = uploaded_file.read()
            if file_bytes:
                with st.spinner("Running inference on uploaded video..."):
                    try:
                        # Prepare request payload
                        files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
                        data = {"gradcam": "true" if gradcam_enabled else "false"}
                        response = requests.post(f"{BACKEND_URL}/predict", files=files, data=data, timeout=60)
                        response.raise_for_status()
                    except Exception as e:
                        st.error(f"Failed to get inference results: {e}")
                    else:
                        result = response.json()
                        # Extract results
                        label = result.get("class") or result.get("label", "")
                        confidence = result.get("confidence")
                        probs = result.get("probabilities")
                        video_url = result.get("video_url")
                        # Display predicted class and confidence
                        if label:
                            conf_pct = confidence * 100 if confidence is not None else None
                            if conf_pct is not None:
                                st.write(f"**Predicted class:** {label} (confidence: {conf_pct:.1f}%)")
                            else:
                                st.write(f"**Predicted class:** {label}")
                        # Display probabilities as bar chart if available
                        if probs:
                            try:
                                # Convert probabilities to a pandas Series or DataFrame for bar_chart
                                if isinstance(probs, dict):
                                    prob_df = pd.DataFrame(list(probs.items()), columns=["Class", "Probability"])
                                    # Convert to percentage:
                                    prob_df["Probability (%)"] = prob_df["Probability"] * 100
                                    prob_df.set_index("Class", inplace=True)
                                    st.bar_chart(prob_df["Probability (%)"])
                                elif isinstance(probs, list):
                                    # If list, assume it's in order of classes (if class labels known, use dict instead)
                                    prob_series = pd.Series([p * 100 for p in probs])
                                    st.bar_chart(prob_series)
                                else:
                                    st.write("Class probabilities:", probs)
                            except Exception as e:
                                st.write("Probabilities:", probs)
                        # Display annotated video if provided
                        if video_url:
                            st.markdown("**Annotated Video:**")
                            st.video(video_url)
                        elif result.get("video_bytes"):
                            # If backend (less likely) returned video content directly (as base64 or bytes)
                            st.markdown("**Annotated Video:**")
                            st.video(result["video_bytes"])
                        # (If neither video_url nor bytes, possibly no video output)
            else:
                st.error("Empty file. Please upload a valid video.")
elif mode == "YouTube Link":
    # Input for YouTube video URL
    youtube_url = st.text_input("Enter YouTube video URL:")
    if youtube_url:
        if st.button("Run Inference"):
            with st.spinner("Fetching video and running inference..."):
                try:
                    data = {"url": youtube_url, "gradcam": "true" if gradcam_enabled else "false"}
                    response = requests.post(f"{BACKEND_URL}/predict", data=data, timeout=60)
                    response.raise_for_status()
                except Exception as e:
                    st.error(f"Inference request failed: {e}")
                else:
                    result = response.json()
                    # Extract and display results similarly to upload case
                    label = result.get("class") or result.get("label", "")
                    confidence = result.get("confidence")
                    probs = result.get("probabilities")
                    video_url = result.get("video_url")
                    if label:
                        conf_pct = confidence * 100 if confidence is not None else None
                        if conf_pct is not None:
                            st.write(f"**Predicted class:** {label} (confidence: {conf_pct:.1f}%)")
                        else:
                            st.write(f"**Predicted class:** {label}")
                    if probs:
                        try:
                            if isinstance(probs, dict):
                                prob_df = pd.DataFrame(list(probs.items()), columns=["Class", "Probability"])
                                prob_df["Probability (%)"] = prob_df["Probability"] * 100
                                prob_df.set_index("Class", inplace=True)
                                st.bar_chart(prob_df["Probability (%)"])
                            elif isinstance(probs, list):
                                prob_series = pd.Series([p * 100 for p in probs])
                                st.bar_chart(prob_series)
                            else:
                                st.write("Class probabilities:", probs)
                        except Exception:
                            st.write("Class probabilities:", probs)
                    if video_url:
                        st.markdown("**Annotated Video:**")
                        st.video(video_url)
                    elif result.get("video_bytes"):
                        st.markdown("**Annotated Video:**")
                        st.video(result["video_bytes"])
    else:
        st.write("Enter a YouTube URL above and click **Run Inference**.")
elif mode == "YouTube + Webcam":
    youtube_url = st.text_input("Enter YouTube video URL:")
    if not st.session_state["start_record"]:
        if st.button("Start Reaction Recording"):
            if not youtube_url or youtube_url.strip() == "":
                st.error("Please provide a YouTube video URL before starting.")
            else:
                vid_id = extract_youtube_id(youtube_url)
                if vid_id is None:
                    st.error("Invalid YouTube URL. Please check and try again.")
                else:
                    st.session_state["yt_video_id"] = vid_id
                    st.session_state["start_record"] = True

    if st.session_state["start_record"]:
        video_id = st.session_state["yt_video_id"]
        gradcam_flag_str = "true" if gradcam_enabled else "false"

        html_code = f"""
        <div>
        <div id="player" style="display:inline-block; width:640px; height:360px; margin-right:10px;"></div>
        <video id="webcamPreview" autoplay muted playsinline
                style="display:inline-block; width:320px; height:240px; background-color:#000;"></video>
        <p id="status-text" style="font-weight:bold; margin:10px 0; color:#000;">
            Initializing camera...
        </p>
        <button id="stop-btn" disabled style="padding:8px 12px; font-size:16px;">
            Stop Recording
        </button>
        </div>
        <script src="https://www.youtube.com/iframe_api"></script>
        <script>
        const videoId = "{video_id}";
        const backendUrl = "{BACKEND_URL}";
        const gradcamFlag = "{gradcam_flag_str}";
        let player;
        let mediaRecorder;
        let recording = false;
        let chunks = [];
        let stream = null;
        let startTime = 0;
        const minDuration = 60000;  // 60s
        let aborted = false;

        function onYouTubeIframeAPIReady() {{
            player = new YT.Player('player', {{
            width: 640, height: 360, videoId: videoId,
            events: {{
                'onReady': onPlayerReady,
                'onStateChange': onPlayerStateChange
            }}
            }});
        }}
        function onPlayerReady(event) {{
            event.target.playVideo();
        }}
        function onPlayerStateChange(event) {{
            if (event.data === YT.PlayerState.PLAYING && !recording) {{
            startWebcamRecording();
            }}
            if (event.data === YT.PlayerState.ENDED) {{
            const elapsed = Date.now() - startTime;
            if (elapsed < minDuration) aborted = true;
            if (recording) stopRecording();
            }}
        }}

        async function startWebcamRecording() {{
            try {{
            stream = await navigator.mediaDevices.getUserMedia({{ video:true, audio:true }});
            }} catch(err) {{
            document.getElementById('status-text').textContent =
                "Error: Cannot access webcam/mic.";
            document.getElementById('status-text').style.color = 'red';
            return;
            }}
            document.getElementById('webcamPreview').srcObject = stream;
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = e => {{ if(e.data.size>0) chunks.push(e.data); }};
            mediaRecorder.onstop = async () => {{
            recording = false;
            stream.getTracks().forEach(t=>t.stop());
            const statusEl = document.getElementById('status-text');
            if (aborted) {{
                statusEl.textContent =
                "Recording too short (<60s). Please re-record.";
                statusEl.style.color = 'red';
                return;
            }}
            statusEl.textContent = "Uploading for inference...";
            const blob = new Blob(chunks, {{ type:'video/webm' }});
            const form = new FormData();
            form.append('file', blob, 'recording.webm');
            form.append('gradcam', gradcamFlag);
            try {{
                const resp = await fetch(backendUrl + "/predict", {{
                method:'POST', body:form
                }});
                if (!resp.ok) throw new Error("Status " + resp.status);
                const result = await resp.json();
                // Display result
                const label = result.class || result.label || "(unknown)";
                const conf = result.confidence;
                if (conf != null) {{
                statusEl.textContent =
                    `Predicted: ${{label}} (Confidence: ${{(conf*100).toFixed(1)}}%)`;
                }} else {{
                statusEl.textContent = `Predicted: ${{label}}`;
                }}
                // Bar chart of probabilities
                if (result.probabilities) {{
                const chartDiv = document.createElement('div');
                chartDiv.innerHTML = "<strong>Class Probabilities:</strong>";
                document.body.appendChild(chartDiv);
                const entries = Array.isArray(result.probabilities)
                    ? result.probabilities.map((p, i) => [`Class ${{i}}`, p])
                    : Object.entries(result.probabilities);
                entries.sort((a,b) => b[1] - a[1]);
                entries.forEach(([cls, p]) => {{
                    const pct = (p*100).toFixed(1) + "%";
                    const line = document.createElement('div');
                    line.style.display = 'flex'; line.style.margin='4px 0';
                    const lbl = document.createElement('span');
                    lbl.textContent = cls;
                    lbl.style.flex = '0 0 100px';
                    const barBg = document.createElement('div');
                    barBg.style.flex='1'; barBg.style.background='#ddd';
                    barBg.style.position='relative'; barBg.style.height='16px';
                    const barFg = document.createElement('div');
                    barFg.style.width=p*100+'%'; barFg.style.height='100%';
                    barFg.style.background='#4caf50';
                    const txt = document.createElement('span');
                    txt.textContent = pct; txt.style.position='absolute';
                    txt.style.right='4px'; txt.style.color='#fff';
                    txt.style.fontSize='12px'; txt.style.lineHeight='16px';
                    barBg.appendChild(barFg);
                    barBg.appendChild(txt);
                    line.appendChild(lbl);
                    line.appendChild(barBg);
                    chartDiv.appendChild(line);
                }});
                }}
                // Annotated video
                if (result.video_url) {{
                const hdr = document.createElement('p');
                hdr.innerHTML = "<strong>Annotated Video:</strong>";
                document.body.appendChild(hdr);
                const vid = document.createElement('video');
                vid.src = result.video_url; vid.controls = true;
                vid.style.maxWidth='100%';
                document.body.appendChild(vid);
                }}
            }} catch(err) {{
                statusEl.textContent = "Inference error: " + err.message;
                statusEl.style.color = 'red';
            }}
            }};
            mediaRecorder.start();
            recording = true;
            startTime = Date.now();
            const stEl = document.getElementById('status-text');
            stEl.textContent = "Recording... 0s (min 60s)";
            setTimeout(() => {{
            document.getElementById('stop-btn').disabled = false;
            }}, minDuration);
            const timer = setInterval(() => {{
            if (!recording) return clearInterval(timer);
            const sec = Math.floor((Date.now() - startTime)/1000);
            stEl.textContent = `Recording... ${{sec}}s` +
                (sec<60 ? " (min 60s)" : " (you can stop now)");
            }}, 1000);
        }}

        function stopRecording() {{
            if (mediaRecorder && recording) mediaRecorder.stop();
            if (player && player.stopVideo) player.stopVideo();
        }}
        document.getElementById('stop-btn').onclick = () => stopRecording();
        </script>
        """

        st.components.v1.html(html_code, height=800)
