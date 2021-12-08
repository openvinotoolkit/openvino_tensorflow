@REM ==============================================================================
@REM Copyright (C) 2021 Intel Corporation
@REM SPDX-License-Identifier: Apache-2.0
@REM ==============================================================================

@echo off
set model_name="yolo_v4"
mkdir temp_build
cd temp_build
python -m venv env
CALL env\Scripts\activate
python -m pip install --upgrade pip
python -m pip install tensorflow==2.5.1 opencv-python==4.5.4.60
python -m pip install pillow numpy matplotlib keras_applications
git clone https://github.com/david8862/keras-YOLOv3-model-set.git tensorflow-yolo-v4
cd tensorflow-yolo-v4
git checkout d38c3d8
git apply ..\..\keras_to_tensorflow.patch
curl https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -o coco.names
curl -L https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -o weights\yolov4.weights
python tools\model_converter\convert.py --yolo4_reorder cfg\yolov4.cfg weights\yolov4.weights weights\yolov4.h5
python tools\model_converter\keras_to_tensorflow.py --input_model weights\yolov4.h5 --output_model=weights\\%model_name% --saved_model
xcopy weights\\%model_name% ..\..\data\\%model_name%  /E /H /Q /S /I
copy coco.names ..\..\data\
cd ..
CALL env\Scripts\deactivate
cd ..
rmdir /s/q temp_build