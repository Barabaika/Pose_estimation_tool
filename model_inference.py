#!/usr/bin/env python

import cv2
import matplotlib.pyplot as plt 
from IPython.display import HTML, display
import numpy as np 
import tensorflow as tf 
import tensorflow_hub as hub
import sys
# custom functions
from inferencs_funcs import *

# Check  for gpu
print('GPU check:', tf.config.list_physical_devices('GPU'))

cyan = (255, 255, 0)
magenta = (255, 0, 255)

EDGE_COLORS = {
    (0, 1): magenta,
    (0, 2): cyan,
    (1, 3): magenta,
    (2, 4): cyan,
    (0, 5): magenta,
    (0, 6): cyan,
    (5, 7): magenta,
    (7, 9): cyan,
    (6, 8): magenta,
    (8, 10): cyan,
    (5, 6): magenta,
    (5, 11): cyan,
    (6, 12): magenta,
    (11, 12): cyan,
    (11, 13): magenta,
    (13, 15): cyan,
    (12, 14): magenta,
    (14, 16): cyan
}

model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures["serving_default"]

WIDTH = HEIGHT = 512

input_files_names_list = sys.argv[1:]

for input_file_name in input_files_names_list:
    if input_file_name == 'test_app_video.mp4':
        print(f"Started with no input video => RUNNING ON TEST VIDEO (test_app_video.mp4)")    
    print(f"****{input_file_name} processing****")
    out_file_name = input_file_name.split('.')[0] + '_output.' + input_file_name.split('.')[1]
    # TODO: SAVE keypoints to json 
    # TODO: import time and print time
    # TODO: convert to GPU and inference from GPU  (for now - to slow about 4 fps) 
    keypoints = run_inference(
        input_file_name, 
        out_file_name
    )

