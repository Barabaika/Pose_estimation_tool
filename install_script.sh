#!/bin/bash

# installing docker 
# curl https://get.docker.com | sh \
#   && sudo systemctl --now enable docker

# apt-get update
# apt-get install -y nvidia-docker2

# (Note that you do not need to install the CUDA Toolkit on the host system, but the NVIDIA driver needs to be installed)
# now it is in dockerfile but may be it is needed to be installed there
# apt-get -y install cuda-drivers 

# STEP1 create image
docker build . -t pose_estimation_image
