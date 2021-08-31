#!/bin/bash
# ==============================================================================
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================

model_name="yolo_v3_darknet_1"
model_source="https://github.com/mystic123/tensorflow-yolo-v3"
mkdir temp_build

if command -v python3.6 >/dev/null 2>&1; then
        echo "python version available 3.6"
        python_binary=python3.6
elif command -v python3.7 >/dev/null 2>&1; then
        echo "python version available3.7"
        python_binary=python3.7
fi

if [ -z "$python_binary" ]; then
    printf "For Tensorflow 1.15.2 conversion of yolov3 darknet model, Python version 3.6 or 3.7 is required. Check your python version.\n"
    exit 1
else
    echo "python version $python_binary"
    cd temp_build
    $python_binary -m venv env
    source env/bin/activate
    $python_binary -m pip install --upgrade pip
    $python_binary -m pip  install tensorflow==1.15.2
    $python_binary -m pip  install pillow
    $python_binary -m pip install numpy==1.19.5
    git clone https://github.com/mystic123/tensorflow-yolo-v3.git
    cd tensorflow-yolo-v3
    git checkout ed60b90
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
    wget https://pjreddie.com/media/files/yolov3.weights
    $python_binary convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3.weights
    cp frozen_darknet_yolov3_model.pb ../../../data/${model_name}.pb
    cp coco.names ../../../data/
    cd ../..
    deactivate
fi
rm -rf temp_build

