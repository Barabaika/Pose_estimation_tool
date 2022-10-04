#!/bin/bash

# STEP1 create image
docker build . -t pose_estimation_image

#TODO: May be restructurize folders?
#TODO: May be place run_pose_estimation.sh to /usr/bin/ for running it from eveywhere
