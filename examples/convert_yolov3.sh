#!/bin/bash
# ==============================================================================
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================

model_name="yolo_v3_darknet"

mkdir temp_build
cd temp_build

# python3 -m venv env
# source env/bin/activate

# pip install --upgrade pip
# pip install tensorflow==1.15.2
# pip install pillow
git clone https://github.com/david8862/keras-YOLOv3-model-set.git tensorflow-yolo-v3
cd tensorflow-yolo-v3
git checkout d38c3d8
patch tools/model_converter/keras_to_tensorflow.py ../../keras_to_tensorflow.patch
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
wget -O weights/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
python tools/model_converter/convert.py cfg/yolov3.cfg weights/yolov3.weights weights/darknet53.h5
python tools/model_converter/keras_to_tensorflow.py --input_model weights/darknet53.h5 --output_model=weights/${model_name}.pb
cp weights/${model_name}.pb ../../data/${model_name}.pb
cp coco.names ../../data/
cd ../..

# deactivate
rm -rf temp_build

