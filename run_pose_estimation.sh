#!/bin/bash

# STEP1: create folder to store there all input images to add it as volume to docker
mkdir tmp_input_files

echo 'Number of input images:'
echo $#

if [ $# -ge 1 ]
then
    cp $@ ./tmp_input_files/
fi

# get absolute path of tmp folder (obligatory for making volume)
tmp_input_files_path= readlink -f ./tmp_input_files
# STEP2: create container + run inference enterypoint
# you can add many images in one input in comand line argument way 

# -v ${tmp_input_files_path}:/usr/src/app/input \
docker run \
    -it pose_estimation_image \
    bash
    # -d --name pose_estimation_container model_inference.py $@

# STEP3: remove tmp files
rm -r tmp_input_files

# STEP2: copy output file from docker container to local dir 
# docker cp <container_name>:/output_file_name ./output_file_name
# docker cp 

# STEP3 stop and remove container
