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
cd temp_build

python3 -m venv env
source env/bin/activate

pip install --upgrade pip
pip install tensorflow==1.15.2
pip install pillow
git clone https://github.com/mystic123/tensorflow-yolo-v3.git
cd tensorflow-yolo-v3
git checkout ed60b90
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
wget https://pjreddie.com/media/files/yolov3.weights
python3 convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3.weights
cp frozen_darknet_yolov3_model.pb ../../data/${model_name}.pb
cp coco.names ../../data/
cd ../..

deactivate
rm -rf temp_build

