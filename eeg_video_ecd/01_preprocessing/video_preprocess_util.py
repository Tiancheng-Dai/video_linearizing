# Helper functions for video_preprocess.py

import glob
import os
import cv2
import numpy as np

FLOW_CLIP = 20.0  # px; standard I3D optical flow clipping range


# ===========================
# Shared resize/pad (keeps rgb and flow spatially aligned)
# ===========================
def _resize_and_pad(frame, size):
    """Resize preserving aspect ratio and pad with black borders to size x size."""
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = round(h * scale), round(w * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top = (size - new_h) // 2
    pad_bottom = size - new_h - pad_top
    pad_left = (size - new_w) // 2
    pad_right = size - new_w - pad_left
    return cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=0)


# ===========================
# RGB stream
# ===========================
def preprocess_frame(frame_bgr, size=224):
    """BGR uint8 frame -> (C, H, W) float32 frame in [-1, 1].

    Converts to RGB, resizes preserving the original aspect ratio, and pads
    with black borders so the result is exactly size x size (letterboxing).
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    padded = _resize_and_pad(frame_rgb, size)
    normalized = padded.astype(np.float32) / 127.5 - 1.0
    return normalized.transpose(2, 0, 1)


# ===========================
# Optical flow stream
# ===========================
def optic_flow(frame_bgr_a, frame_bgr_b, size=224, downsample=4):
    """Two consecutive BGR uint8 frames -> (2, H, W) float32 flow in [-1, 1].

    Resizes/pads both frames identically to preprocess_frame (so the flow
    stream stays pixel-aligned with the RGB stream), then downsamples by
    `downsample` before running Farneback -- this denoises small
    camera/compression jitter and gives Farneback's window a proportionally
    larger receptive field, which reads as much cleaner flow. The resulting
    field is upsampled back to size x size (scaling displacement by the same
    factor) before clipping to +/-FLOW_CLIP px and normalizing to [-1, 1].
    """
    gray_a = _resize_and_pad(cv2.cvtColor(frame_bgr_a, cv2.COLOR_BGR2GRAY), size)
    gray_b = _resize_and_pad(cv2.cvtColor(frame_bgr_b, cv2.COLOR_BGR2GRAY), size)

    small_size = size // downsample
    small_a = cv2.resize(gray_a, (small_size, small_size), interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(gray_b, (small_size, small_size), interpolation=cv2.INTER_AREA)

    small_flow = cv2.calcOpticalFlowFarneback(
        small_a, small_b, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)  # (h, w, 2) px displacement at small_size

    flow = cv2.resize(small_flow, (size, size), interpolation=cv2.INTER_LINEAR) * downsample

    normalized = (np.clip(flow, -FLOW_CLIP, FLOW_CLIP) / FLOW_CLIP).astype(np.float32)
    return normalized.transpose(2, 0, 1)


# ===========================
# Frame reconstruction (undo preprocess_frame/optic_flow's [-1, 1] encoding)
# ===========================
def _rgb_npy_to_bgr(path):
    """(3, H, W) float32 [-1, 1] .npy -> (H, W, 3) uint8 BGR image."""
    frame = np.load(path)
    frame = ((frame + 1.0) * 127.5).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def _flow_npy_to_bgr(path):
    """(2, H, W) float32 [-1, 1] .npy -> (H, W, 3) uint8 BGR color-wheel image.

    Hue encodes flow direction, brightness encodes flow magnitude.
    """
    flow = np.load(path)
    dx, dy = flow[0] * FLOW_CLIP, flow[1] * FLOW_CLIP
    magnitude, angle = cv2.cartToPolar(dx, dy)
    hsv = np.zeros((*dx.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 90 / np.pi  # radians [0, 2pi) -> hue [0, 180)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude / FLOW_CLIP * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ===========================
# Side-by-side sample video (rgb | flow)
# ===========================
def write_side_by_side_video(rgb_frame_dir, flow_frame_dir, save_path, fps):
    """Reassembles matching rgb + flow frames into one mp4, RGB left / flow right."""
    rgb_files = sorted(glob.glob(os.path.join(rgb_frame_dir, '*.npy')))
    flow_files = sorted(glob.glob(os.path.join(flow_frame_dir, '*.npy')))
    num_frames = min(len(rgb_files), len(flow_files))  # flow has one fewer frame

    writer = None
    for i in range(num_frames):
        side_by_side = np.hstack([
            _rgb_npy_to_bgr(rgb_files[i]),
            _flow_npy_to_bgr(flow_files[i]),
        ])
        if writer is None:
            h, w = side_by_side.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
        writer.write(side_by_side)
    writer.release()
    return num_frames
