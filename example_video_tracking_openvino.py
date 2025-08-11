"""
Usage: python __file__ path_to_decoder.xml path_to_encoder.xml

Modify device from GPU to CPU in the script to use CPU
"""

import cv2
import sys
import torch
import numpy as np
import pickle
from cap_from_youtube import cap_from_youtube

import tapnet.utils as utils
from tapnet.tapir_inference import OVTapirInference

ov_device = "GPU"

# tapvid_davis dataset: `curl -O "https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip`
with open("tapvid_davis/tapvid_davis.pkl", "rb") as f:
    data = pickle.load(f)

video = data["dog"]["video"]

# Initialize model
model_path = sys.argv[1]
encoder_path = sys.argv[2]
tapir = OVTapirInference(model_path, encoder_path, ov_device)
input_size = tapir.input_resolution[0]
num_points = tapir.num_points
num_iters = tapir.num_iters
print(num_points)

# Initialize query features
query_points = utils.sample_grid_points(input_size, input_size, num_points)
point_colors = utils.get_colors(num_points)

track_length = 30
tracks = np.zeros((num_points, track_length, 2), dtype=object)
cv2.namedWindow('video', cv2.WINDOW_NORMAL)

for i, frame in enumerate(video):
    if i == 0:
        tapir.set_points(frame, query_points)
    # Run the model
    points, visibles = tapir.forward(frame)

    # Record visible points
    tracks = np.roll(tracks, 1, axis=1)
    tracks[visibles, 0] = points[visibles]
    tracks[~visibles, 0] = -1

    # Draw the results
    frame = utils.draw_tracks(frame[:,:,::-1], tracks, point_colors)
    frame = utils.draw_points(frame, points, visibles, point_colors)
    cv2.imshow('video', frame)

    # out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
