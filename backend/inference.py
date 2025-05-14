# backend/inference.py
import torch, cv2
import numpy as np
from torch.nn.functional import sigmoid
import os
import uuid
# Load model definitions from the repository
from models.simple3dlstm import Simple3DLSTM
from models.pretrained_r3d import PretrainedR3D

# Dictionary to cache loaded models in memory (to avoid re-loading on every request)
_loaded_models = {}

# Assume binary classification: class 1 = ASD Positive, class 0 = ASD Negative
CLASS_NAMES = ["ASD Negative", "ASD Positive"]

def load_model(model_name):
    """Load the model architecture and weights if not already loaded."""
    if model_name in _loaded_models:
        return _loaded_models[model_name]
    # Instantiate model architecture
    if model_name.lower().startswith("simple3dlstm"):
        # Example: model_name could be "simple3dlstm_fold1"
        model = Simple3DLSTM(max_frames=60, target_size=(112,112))
    elif model_name.lower().startswith("r3d") or model_name.lower().startswith("pretrainedr3d"):
        model = PretrainedR3D()
    else:
        raise ValueError(f"Unknown model name {model_name}")
    # Load weights
    weights_path = os.path.join("backend/saved_models", model_name + ".pth")
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()  # set to evaluation mode
    _loaded_models[model_name] = model
    return model

def preprocess_video(video_path, max_frames=60, target_size=(112, 112)):
    """
    Load video frames from the given path, sample up to max_frames frames, 
    and resize/normalize them into a tensor suitable for the model.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # default to 30 if not available
    frames = []

    # Determine frame indices to sample
    if total_frames <= max_frames or max_frames is None:
        frame_indices = list(range(total_frames))
    else:
        # Sample evenly spaced frames (to cover the video uniformly)
        frame_indices = np.linspace(0, total_frames-1, num=max_frames, dtype=int).tolist()

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame to target size
        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert BGR (OpenCV) to RGB
        frames.append(frame)
    cap.release()

    # Pad frames if fewer than max_frames
    if len(frames) < max_frames:
        pad_count = max_frames - len(frames)
        pad_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)  # black frames
        frames.extend([pad_frame] * pad_count)

    # Convert to numpy and normalize
    video_array = np.array(frames, dtype=np.float32) / 255.0  # shape (T, H, W, C)
    # Rearrange to (C, T, H, W) for PyTorch
    video_tensor = torch.from_numpy(video_array).permute(3, 0, 1, 2).unsqueeze(0)  # add batch dim
    # Normalize with ImageNet mean/std (as used in training)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None, None]
    video_tensor = (video_tensor - mean) / std
    return video_tensor, fps

def run_inference(video_path, model_name):
    # Load or retrieve the model
    model = load_model(model_name)
    # Preprocess video into a tensor
    video_tensor, fps = preprocess_video(video_path)
    # Run model prediction (assuming model outputs a single logit for the "Positive" class)
    with torch.no_grad():
        logit = model(video_tensor)             # model output (logit):contentReference[oaicite:5]{index=5}
        prob = sigmoid(logit)[0].item()         # convert to probability
    # Determine label based on threshold 0.5 (or use calibrated threshold if available)
    label_idx = 1 if prob >= 0.5 else 0
    label_name = CLASS_NAMES[label_idx]
    confidence = prob if label_idx == 1 else (1 - prob)  # confidence of the predicted class

    # Annotate video frames with prediction (since classification is per-video, we overlay overall result)
    annotated_path = os.path.join("backend/static", f"annotated_{uuid.uuid4()}.mp4")
    _annotate_video(video_path, label_name, prob, annotated_path, fps)

    return {"label": label_name, "confidence": round(confidence, 4), "annotated_video": annotated_path}

def _annotate_video(video_path, label, probability, output_path, fps):
    """Generate a copy of the video with the predicted label and confidence overlaid on each frame."""
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    text = f"{label} ({probability*100:.1f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (0, 255, 0) if "Positive" in label else (0, 0, 255)  # green for positive, red for negative
    thickness = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Overlay text on frame (top-left corner)
        cv2.putText(frame, text, (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
        out.write(frame)
    cap.release()
    out.release()
