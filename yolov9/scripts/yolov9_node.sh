#!/bin/bash

source /home/super/anaconda3/etc/profile.d/conda.sh
conda activate yolov9

python /home/super/catkin_ws/src/screw_ros/yolov9/scripts/detect_ros.py "$@"
