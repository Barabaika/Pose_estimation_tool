#!/usr/bin/env python

import os
import pickle
import time

# custom functions
from inference_funcs import *


with open('edge_colors.pickle', 'rb') as handle:
    EDGE_COLORS = pickle.load(handle)

model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures["serving_default"]

# WIDTH = HEIGHT = 512

input_files_names_list = os.listdir('./tmp_input_files')

for input_file_name in input_files_names_list:
    if input_file_name == 'test_app_video.mp4':
        print(f"Started with no input video => RUNNING ON TEST VIDEO (test_app_video.mp4)")    
    print(f"****{input_file_name} processing****")
    out_file_name = '/usr/src/app/tmp_input_files/' + input_file_name.split('.')[0] + '_output.' + input_file_name.split('.')[1]
    keypoints = run_inference(
        input_file_name, 
        out_file_name,
        movenet,
        EDGE_COLORS
    )

