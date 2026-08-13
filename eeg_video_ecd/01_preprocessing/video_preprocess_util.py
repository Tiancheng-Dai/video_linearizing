# Helper functions for video_preprocess.py

import glob
import os
import cv2
import numpy as np


def preprocess_frame(frame_bgr, size=224):
    """BGR uint8 frame -> (C, H, W) float32 frame in [-1, 1].

    Converts to RGB, resizes preserving the original aspect ratio, and pads
    with black borders so the result is exactly size x size (letterboxing).
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    h, w = frame_rgb.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = round(h * scale), round(w * scale)
    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top = (size - new_h) // 2
    pad_bottom = size - new_h - pad_top
    pad_left = (size - new_w) // 2
    pad_right = size - new_w - pad_left
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

    normalized = padded.astype(np.float32) / 127.5 - 1.0
    return normalized.transpose(2, 0, 1)


def write_video_from_frame_dir(frame_dir, save_path, fps):
    """Reassembles the ordered *.npy frames in frame_dir into an mp4 file."""
    frame_files = sorted(glob.glob(os.path.join(frame_dir, '*.npy')))
    writer = None
    for frame_file in frame_files:
        frame = np.load(frame_file)  # (C, H, W) float32 in [-1, 1]
        frame = ((frame + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        frame = frame.transpose(1, 2, 0)  # HWC RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if writer is None:
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
        writer.write(frame_bgr)
    writer.release()
