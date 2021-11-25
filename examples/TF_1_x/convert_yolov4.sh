#!/bin/bash
# ==============================================================================
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================

model_name="yolo_v4"

mkdir temp_build
cd temp_build
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install tensorflow==2.7.0 opencv-python==4.2.0.32
python3 -m pip install pillow numpy matplotlib keras_applications
git clone https://github.com/david8862/keras-YOLOv3-model-set.git tensorflow-yolo-v4
cd tensorflow-yolo-v4
git checkout d38c3d8
patch tools/model_converter/keras_to_tensorflow.py ../../keras_to_tensorflow.patch
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
wget -O weights/yolov4.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
python3 tools/model_converter/convert.py --yolo4_reorder cfg/yolov4.cfg weights/yolov4.weights weights/yolov4.h5
python3 tools/model_converter/keras_to_tensorflow.py --input_model weights/yolov4.h5 --output_model=weights/${model_name}.pb
cp -r weights/${model_name}.pb ../../../data/${model_name}.pb
cp coco.names ../../../data/
cd ../../..
deactivate
rm -rf temp_build

