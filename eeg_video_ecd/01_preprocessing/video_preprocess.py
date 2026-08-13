# Preprocessing traffic videos into per-frame arrays ready for the I3D model.

import argparse
import glob
import os
import cv2
import numpy as np
from tqdm import tqdm

from video_preprocess_util import preprocess_frame, write_video_from_frame_dir

TARGET_SIZE = 224

# ===========================
# Setup
# ===========================
script_dir = os.path.dirname(os.path.abspath(__file__))
default_project_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default=default_project_dir, type=str)
args = parser.parse_args()

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
# 3) Resize + extract frames, save one file per frame per video
# ===========================
sample_video_out_dir = None
sample_fps = 25.0
for video_path in tqdm(video_paths, desc='Preprocessing videos'):
    # Keep the original video's index (its filename stem) instead of renumbering it
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    video_out_dir = os.path.join(out_root, video_stem)
    os.makedirs(video_out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_order = 0
    while frame_order < num_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        # Resize to 224x224, preserve RGB, pad to keep the 1080p aspect ratio
        i3d_frame = preprocess_frame(frame_bgr, size=TARGET_SIZE)
        # Name each frame by the order it appears in the video
        frame_path = os.path.join(video_out_dir, f'{frame_order:04d}.npy')
        np.save(frame_path, i3d_frame)
        frame_order += 1
    cap.release()

    # Remember the first video processed to build the sample video from
    if sample_video_out_dir is None:
        sample_video_out_dir = video_out_dir
        sample_fps = fps

# ===========================
# 4) Compile one video's preprocessed frames back into an mp4
# ===========================
sample_save_path = os.path.join(args.project_dir, 'sample_video.mp4')
write_video_from_frame_dir(sample_video_out_dir, sample_save_path, sample_fps)
print(f'Sample video written to {sample_save_path}')
