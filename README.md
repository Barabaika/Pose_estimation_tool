<h1 align="center">
  Pose_estimation_tool
  <br>
</h1>

Tool for Pose estimation from .mp4 video.
Based on Movenet model 
(https://tfhub.dev/google/movenet/singlepose/thunder/4)

## 📝 Requirements

+ __Docker__ (Docker Desktop or Docker Engine (https://docs.docker.com/desktop/))
+ __Git__ (https://github.com/git-guides/install-git)

## ⚙️ Installation

To install `sc-intregnet` run in terminal:

```bash
git clone https://github.com/Barabaika/Pose_estimation_tool.git
cd Pose_estimation_tool && ./install.sh
```

## 🚀 Usage

### To run at test video:

Go to Pose_estimation_tool folder, and then:

```bash
./run_pose_estimation.sh

```

### To run on your own videos:

Go to Pose_estimation_tool folder, and then:

```bash
./run_pose_estimation.sh <abs_path_to_input_video.mp4>*

```
*- you can add multiple paths

### Results

Results will be stored at Pose_estimation_tool/Pose_estim_result/

From each input video it would be generated:

+ __<input_video_name>_output.mp4__ (video with keypoints and edges)
+ __<input_video_name>_output.json__ ({'poses':{'pose_1':list_keypoints, ..., 'pose_n':list_keypoints}})
