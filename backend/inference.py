# backend/inference.py

import os
import uuid
import cv2
import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# import your model definitions
from models.simple3dlstm import Simple3DLSTM
from models.pretrained_r3d import PretrainedR3D

# where your saved weights live
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
# static output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "static")

# class names for binary classification
CLASS_NAMES = ["ASD Negative", "ASD Positive"]

# cache loaded models in memory
_loaded_models = {}


def load_model(model_name: str):
    """
    Instantiate the model architecture and load its weights from disk.
    Caches loaded models to speed up repeated inference calls.
    """
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    # decide which architecture to use
    name = model_name.lower()
    if name.startswith("simple3dlstm"):
        model = Simple3DLSTM(max_frames=60, target_size=(112, 112))
    elif name.startswith("pretrainedr3d") or name.startswith("r3d"):
        model = PretrainedR3D()
    else:
        raise ValueError(f"Unrecognized model name '{model_name}'")

    # load weights
    weights_path = os.path.join(WEIGHTS_DIR, model_name + ".pth")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weight file not found: {weights_path}")
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    _loaded_models[model_name] = model
    return model


def preprocess_video(video_path: str,
                     max_frames: int = 60,
                     target_size: tuple = (112, 112)):
    """
    Read video frames, sample up to max_frames evenly,
    resize to target_size, normalize, and return a tensor
    of shape (1, C, T, H, W), plus the video's original FPS.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # choose frame indices
    if total <= max_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num=max_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        # resize & convert BGR→RGB
        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    # pad if too few frames
    if len(frames) < max_frames:
        pad_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        frames += [pad_frame] * (max_frames - len(frames))

    # to float32 [0,1]
    video_np = np.array(frames, dtype=np.float32) / 255.0  # shape (T,H,W,3)
    # transpose to (C,T,H,W) and add batch dim
    video_t = torch.from_numpy(video_np).permute(3, 0, 1, 2).unsqueeze(0)

    # normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None, None]
    video_t = (video_t - mean) / std

    return video_t, fps


def _get_target_layer(model: torch.nn.Module):
    """
    Heuristic: return the best layer for GradCAM:
      - If ResNet3D style: use model.layer4
      - If Simple3DLSTM: use model.conv3 (last conv)
      - Otherwise scan for last Conv3d
    """
    if hasattr(model, "layer4"):
        return model.layer4
    if hasattr(model, "conv3"):
        return model.conv3
    # fallback: find last Conv3d
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv3d):
            return m
    raise RuntimeError("No Conv3d layer found for GradCAM")


def generate_gradcam_frames(model: torch.nn.Module,
                            video_tensor: torch.Tensor,
                            class_idx: int):
    """
    Computes a grayscale CAM map for each frame.
    Returns a numpy array shape (T, H, W) with values [0..1].
    """
    target_layer = _get_target_layer(model)

    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
        use_cuda=torch.cuda.is_available()
    )
    targets = [ClassifierOutputTarget(class_idx)]
    # result is (1, T, H, W)
    grayscale_cam = cam(input_tensor=video_tensor, targets=targets)[0]
    # clamp to [0,1]
    grayscale_cam = np.clip(grayscale_cam, 0.0, 1.0)
    return grayscale_cam  # shape (T, H, W)


def write_video_with_heatmaps(heatmaps: np.ndarray,
                              src_path: str,
                              out_path: str,
                              fps: float):
    """
    Overlay each frame's heatmap (H,W) onto the original video frames
    and write the result to out_path.
    """
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {src_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    t = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask = heatmaps[t]  # shape (h_mask, w_mask)
        # resize mask to frame size
        mask_resized = cv2.resize(mask, (width, height))
        heatmap = cv2.applyColorMap(
            np.uint8(mask_resized * 255),
            cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)
        writer.write(overlay)
        t += 1

    cap.release()
    writer.release()


def write_video_with_text(src_path: str,
                          label: str,
                          prob: float,
                          out_path: str,
                          fps: float):
    """
    Overlay prediction text on each frame of the source video.
    """
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {src_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    text = f"{label} ({prob * 100:.1f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (0, 255, 0) if label.endswith("Positive") else (0, 0, 255)
    thickness = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(
            frame, text, (10, 30),
            font, font_scale, color, thickness, cv2.LINE_AA
        )
        writer.write(frame)

    cap.release()
    writer.release()


def run_inference(video_path: str,
                  model_name: str,
                  gradcam: bool = False):
    """
    End-to-end inference:
      1. Load model
      2. Preprocess video → tensor
      3. Forward pass → probability
      4. Depending on gradcam flag, generate and save either:
         - a heatmap-overlay video
         - a simple text-overlay video
      5. Return dict with label, confidence, and annotated_video path
    """
    model = load_model(model_name)
    video_tensor, fps = preprocess_video(video_path)

    with torch.no_grad():
        logit = model(video_tensor)            # shape (1,)
        prob = torch.sigmoid(logit)[0].item()  # [0..1]

    label_idx = 1 if prob >= 0.5 else 0
    label = CLASS_NAMES[label_idx]
    confidence = prob if label_idx == 1 else (1 - prob)

    # choose output path
    tag = "gradcam" if gradcam else "text"
    out_fname = f"{tag}_{uuid.uuid4()}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_fname)

    if gradcam:
        heatmaps = generate_gradcam_frames(model, video_tensor, class_idx=label_idx)
        write_video_with_heatmaps(heatmaps, video_path, out_path, fps)
    else:
        write_video_with_text(video_path, label, prob, out_path, fps)

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "annotated_video": out_path
    }
