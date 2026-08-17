# Preprocessing traffic videos into per-frame arrays ready for the I3D model.

import argparse
import glob
import os
import cv2
import numpy as np
from tqdm import tqdm

from video_preprocess_util import preprocess_frame, optic_flow

TARGET_SIZE = 224

# ===========================
# Setup
# ===========================
script_dir = os.path.dirname(os.path.abspath(__file__))
default_project_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default=default_project_dir, type=str)
parser.add_argument('--rgb', action='store_true', help='Preprocess the RGB stream')
parser.add_argument('--flow', action='store_true', help='Preprocess the optical flow stream')
args = parser.parse_args()
if not args.rgb and not args.flow:
    parser.error('Specify --rgb and/or --flow')

data_dir = os.path.join(args.project_dir, 'video_dataset', 'source_data')
out_root = os.path.join(args.project_dir, 'video_dataset', 'preprocessed_data')
os.makedirs(out_root, exist_ok=True)

# ===========================
# 1) Fetch all traffic*.mp4 videos
# ===========================
video_paths = sorted(glob.glob(os.path.join(data_dir, 'traffic*.mp4')))
if not video_paths:
    raise FileNotFoundError(f'No traffic*.mp4 videos found in {data_dir}')
print(f'Found {len(video_paths)} videos in {data_dir}')

# ===========================
# 2) Find the common frame count
# ===========================
# Every output video must have the same number of frames, so use the
# dataset-wide minimum: every clip gets truncated, none get padded.
frame_counts = []
for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)
    frame_counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
num_frames = min(frame_counts)
print(f'Using {num_frames} frames per video')

# ===========================
# 3) RGB stream: resize + extract frames, save one file per frame per video
# ===========================
if args.rgb:
    for video_path in tqdm(video_paths, desc='Preprocessing RGB stream'):
        # Keep the original video's index (its filename stem) instead of renumbering it
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        video_out_dir = os.path.join(out_root, video_stem, 'rgb')
        os.makedirs(video_out_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_order = 0
        while frame_order < num_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            # Resize to 224x224, preserve RGB, pad to keep the 1080p aspect ratio
            i3d_frame = preprocess_frame(frame_bgr, size=TARGET_SIZE)
            # Name each frame by the order it appears in the video
            frame_path = os.path.join(video_out_dir, f'rgb_{frame_order:04d}.npy')
            np.save(frame_path, i3d_frame)
            frame_order += 1
        cap.release()
    print(f'RGB stream done: {len(video_paths)} videos -> {out_root}/<video>/rgb/')

# ===========================
# 4) Flow stream: dense optical flow between consecutive frames
# ===========================
if args.flow:
    for video_path in tqdm(video_paths, desc='Preprocessing flow stream'):
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        video_out_dir = os.path.join(out_root, video_stem, 'flow')
        os.makedirs(video_out_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        ok, prev_frame = cap.read()
        frame_order = 0
        # num_frames frames -> num_frames - 1 consecutive-pair flow fields
        while ok and frame_order < num_frames - 1:
            ok, next_frame = cap.read()
            if not ok:
                break
            flow_frame = optic_flow(prev_frame, next_frame, size=TARGET_SIZE)
            frame_path = os.path.join(video_out_dir, f'flow_{frame_order:04d}.npy')
            np.save(frame_path, flow_frame)
            prev_frame = next_frame
            frame_order += 1
        cap.release()
    print(f'Flow stream done: {len(video_paths)} videos -> {out_root}/<video>/flow/')
