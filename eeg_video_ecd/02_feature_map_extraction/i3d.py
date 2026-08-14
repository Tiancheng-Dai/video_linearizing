"""import the i3d and save feature maps"""

import argparse
import glob
import os
import sys

import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
default_project_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, default_project_dir)

from kinetics_i3d_pytorch.src.i3dpt import I3D

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default=default_project_dir, type=str)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--stream', default='rgb', type=str)
args = parser.parse_args()

data_dir = args.data_dir or os.path.join(args.project_dir, 'video_dataset', 'preprocessed_data')
output_dir = args.output_dir or os.path.join(args.project_dir, 'feature_maps', 'feature_map_raw')
os.makedirs(output_dir, exist_ok=True)

# ===========================
# Load frames
# ===========================
# Each video is a subfolder of ordered per-frame (C, H, W) float32 .npy arrays
# (produced by video_preprocess.py), already in I3D's expected [-1, 1] range.
video_dirs = sorted(d for d in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(d))
if not video_dirs:
    raise FileNotFoundError(f'No preprocessed video folders found in {data_dir}')

# ===========================
# Run model
# ===========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.stream == 'rgb':
    model = I3D(num_classes=400, modality='rgb')
elif args.stream == 'flow':
    model = I3D(num_classes=400, modality='flow')
else:
    raise NotImplementedError(f'{args.stream} is not supported for Inflated 3D')

weights_path = os.path.join(args.project_dir, 'kinetics_i3d_pytorch', 'model', 'model_rgb.pth')
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device).eval()

# I3D's forward() only returns the final 400-way classification logits, not
# feature maps. Grab the pooled Mixed_5c embedding instead via a forward hook
# on avg_pool (the layer feeding the classification head).
features = {}


def _capture_features(module, inp, out):
    features['avg_pool'] = out.detach()


model.avg_pool.register_forward_hook(_capture_features)

for video_dir in video_dirs:
    video_name = os.path.basename(video_dir)
    frame_files = sorted(glob.glob(os.path.join(video_dir, '*.npy')))

    frames = np.stack([np.load(f) for f in frame_files], axis=1)  # (C, T, H, W)
    x = torch.from_numpy(frames).unsqueeze(0).to(device)  # (1, C, T, H, W)

    with torch.no_grad():
        model(x)
    feature_map = features['avg_pool'].squeeze(0).cpu().numpy()  # (1024, T', 1, 1)

    # ===========================
    # Save feature maps (keep the original video's index in the filename)
    # ===========================
    np.save(os.path.join(output_dir, f'{video_name}.npy'), feature_map)
    print(f'{video_name}: saved feature map with shape {feature_map.shape}')
