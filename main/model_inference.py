#!/usr/bin/env python

import os
import pickle
import time
import json
import time

# custom functions
from inference_funcs import *


with open("help_files/edge_colors.pickle", "rb") as handle:
    EDGE_COLORS = pickle.load(handle)

model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = model.signatures["serving_default"]

# WIDTH = HEIGHT = 512

input_files_names_list = os.listdir("./tmp_input_files")

for input_file_name in input_files_names_list:

    start_time = time.time()

    if input_file_name == "test_app_video.mp4":
        print(
            f"Started with no input video => RUNNING ON TEST VIDEO (test_app_video.mp4)"
        )
    print(f"****{input_file_name} processing****")

    out_video_name = (
        "/usr/src/app/tmp_input_files/"
        + input_file_name.split(".")[0]
        + "_output."
        + input_file_name.split(".")[1]
    )
    out_json_name = (
        "/usr/src/app/tmp_input_files/"
        + input_file_name.split(".")[0]
        + "_output."
        + "json"
    )

    input_file_path = "/usr/src/app/tmp_input_files/" + input_file_name

    keypoints_dict = run_inference(
        input_file_path, out_video_name, movenet, EDGE_COLORS, use_cropping=True, FPS=20
    )

    print("--- %s seconds ---" % (time.time() - start_time))

    res_key_dict = {"frame_poses": keypoints_dict}

    jsonString = json.dumps(res_key_dict)
    jsonFile = open(out_json_name, "w")
    jsonFile.write(jsonString)
    jsonFile.close()
