#!/bin/bash


# STEP1: create container + run inference enterypoint
# you can add many images in one input in comand line argument way 
docker run -d --name pose_estimation_container model_inference.py $@

# STEP2: copy output file from docker container to local dir 
# docker cp <container_name>:/output_file_name ./output_file_name
# docker cp 

# STEP3 stop and remove container
