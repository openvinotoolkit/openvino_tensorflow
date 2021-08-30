#!/bin/bash
# ==============================================================================
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================

model_name="yolo_v3_darknet"
input_node="inputs"
output_node="output_boxes"
model_source="https://github.com/mystic123/tensorflow-yolo-v3"

mkdir temp_build

ver36=$(ls -1 /usr/bin/python* | grep -i 3.6)
if [ -z "$ver36" ]
then
        echo "Python3.6 is not available"
else
        cd temp_build
        python3.6 -m venv env
        source env/bin/activate
        python3.6 -m pip install --upgrade pip
        python3.6 -m pip install tensorflow==1.15.2
        python3.6 -m pip install pillow
        git clone https://github.com/mystic123/tensorflow-yolo-v3.git
        cd tensorflow-yolo-v3
        git checkout ed60b90
        wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
        wget https://pjreddie.com/media/files/yolov3.weights
        python3.6 convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3.weights
        cp frozen_darknet_yolov3_model.pb ../../data/${model_name}.pb
        cp coco.names ../../data/
        cd ../..
        deactivate
fi
rm -rf temp_build

