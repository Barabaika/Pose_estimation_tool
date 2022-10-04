#!/bin/bash

# STEP1: create folder to store there all input images to add it as volume to docker
mkdir tmp_input_files

echo 'Number of input images:'
echo $#

if [ $# -ge 1 ]
then
    cp $@ ./tmp_input_files/
else
# if no videos passed as argument - add test_video for demo 
    cp test_app_video.mp4 ./tmp_input_files/
fi

# get absolute path of tmp folder (obligatory for making volume)
tmp_input_files_path=$(readlink -f ./tmp_input_files)

# STEP2: create container + run inference enterypoint
# you can add many images in one input in comand line argument way 
docker run \
    --name pose_estimation_container --rm \
    -v ${tmp_input_files_path}:/usr/src/app/tmp_input_files/ \
    -it pose_estimation_image bash \
     model_inference.py $@

# STEP3: copy outputs to result folder and remove tmp folder
mkdir ../Pose_estim_result/
cp ./tmp_input_files/*_output.mp4 ../Pose_estim_result/
rm -r tmp_input_files
