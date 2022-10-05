#!/usr/bin/env python

import cv2
from IPython.display import HTML, display
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os

# Custom func for cropping
from cropping import *

# script with functions used in model inference.py


def draw(frame, keypoints, EDGE_COLORS, size, threshold=0.11):
    """
    Draws the keypoints and edges
    """
    # Draw the keypoints and get the denormalized coordinates
    denormalized_coordinates = draw_keypoints(frame, keypoints, size, threshold)
    # Draw the edges
    draw_edges(denormalized_coordinates, frame, EDGE_COLORS, threshold)


def draw_keypoints(frame, keypoints, size, threshold=0.11):
    """Draws the keypoints on a image frame"""

    # Denormalize the coordinates : multiply the normalized coordinates by the input_size(width,height)
    denormalized_coordinates = np.squeeze(np.multiply(keypoints, [size[0], size[1], 1]))
    # Iterate through the points
    for keypoint in denormalized_coordinates:
        # Unpack the keypoint values : y, x, confidence score
        keypoint_y, keypoint_x, keypoint_confidence = keypoint
        if keypoint_confidence > threshold:
            """ "
            Draw the circle
            Note : A thickness of -1 px will fill the circle shape by the specified color.
            """
            cv2.circle(
                img=frame,
                center=(int(keypoint_x), int(keypoint_y)),
                radius=4,
                color=(255, 0, 0),
                thickness=-1,
            )
    return denormalized_coordinates


def draw_edges(denormalized_coordinates, frame, edges_colors, threshold=0.11):
    """
    Draws the edges on a image frame
    """

    # Iterate through the edges
    for edge, color in edges_colors.items():
        # Get the dict value associated to the actual edge
        p1, p2 = edge
        # Get the points
        y1, x1, confidence_1 = denormalized_coordinates[p1]
        y2, x2, confidence_2 = denormalized_coordinates[p2]
        # Draw the line from point 1 to point 2, the confidence > threshold
        if (confidence_1 > threshold) & (confidence_2 > threshold):
            cv2.line(
                img=frame,
                pt1=(int(x1), int(y1)),
                pt2=(int(x2), int(y2)),
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,  # Gives anti-aliased (smoothed) line which looks great for curves
            )


def load_video(input_video_path):
    """
    Loads the video and return its details
    """

    # Load the video
    video = cv2.VideoCapture(input_video_path)
    # Get the frame count
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # Display parameter
    print(f"Frame count: {frame_count}")

    # Get the initial shape (width, height)
    initial_shape = []
    initial_shape.append(int(video.get(cv2.CAP_PROP_FRAME_WIDTH)))
    initial_shape.append(int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # return video, frame_count, output_frames, initial_shape
    return video, frame_count, initial_shape


def run_inference(
    input_video_path,
    out_video_path,
    model_func,
    EDGE_COLORS,
    use_cropping=True,
    FPS=20,
    INFERENCE_SIZE=(256, 256),
):
    """
    Runs inferences then starts the main loop for each frame
    """

    # Load the video
    video, frame_count, initial_shape = load_video(input_video_path)

    # for cropping:
    if use_cropping:
        crop_region = init_crop_region(INFERENCE_SIZE[0], INFERENCE_SIZE[1])

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        out_video_path, fourcc, float(FPS), (initial_shape[0], initial_shape[1])
    )
    # Create keypoints result dict for storing keypoints from each frame
    keypoints_dict = {}

    # Create frame counter for storing keypoints to dict
    frame_counter = -1
    # Loop while the video is opened
    while video.isOpened():

        frame_counter += 1

        # Capture the frame
        ret, frame = video.read()

        # Exit if the frame is empty
        if frame is None:
            break

        # Retrieve the frame index
        current_index = video.get(cv2.CAP_PROP_POS_FRAMES)

        # Copy the frame
        image = frame.copy()
        image = cv2.resize(image, INFERENCE_SIZE)
        # Resize to the target shape and cast to an int32 vector
        input_image = tf.cast(
            tf.image.resize_with_pad(image, INFERENCE_SIZE[0], INFERENCE_SIZE[1]),
            dtype=tf.int32,
        )

        # inference variant without cropping
        if not use_cropping:
            # Create a batch (input tensor)
            input_image = tf.expand_dims(input_image, axis=0)

            # Perform inference
            results = model_func(input_image)

            # initial shape of res is (1,1,17,3)
            keypoints = results["output_0"][0][0]

        # inference variant with cropping
        else:
            crop_size = INFERENCE_SIZE
            keypoints = inference_with_cropping(
                model_func, input_image, crop_region, crop_size
            )
            keypoints = keypoints[0][0]

        # Draw the results to frame
        draw(image, keypoints, EDGE_COLORS, INFERENCE_SIZE, threshold=0.11)

        # Denormalizing resulted keypoints to initial scale and crop 3 column with scores
        keypoints = np.multiply(keypoints[:, 0:2], [initial_shape[0], initial_shape[1]])
        # Adding to resulted dict
        keypoints_dict["pose_" + str(frame_counter)] = keypoints.tolist()

        # Get the output frame : reshape to the original size
        frame_rgb = cv2.cvtColor(
            cv2.resize(
                image,
                (initial_shape[0], initial_shape[1]),
                interpolation=cv2.INTER_LANCZOS4,
            ),
            cv2.COLOR_BGR2RGB,  # OpenCV processes BGR images instead of RGB
        )

        # Add resulted frame to the video_writer
        video_writer.write(frame_rgb)

    # Release the object
    video_writer.release()

    print("Completed !")

    return keypoints_dict
