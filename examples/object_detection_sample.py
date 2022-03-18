# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# ==============================================================================
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
# Modified from tensorflow object detection examples:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/label_image.py
# https://github.com/mystic123/tensorflow-yolo-v3/blob/master/utils.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# Enable these variables for runtime inference optimizations
os.environ["OPENVINO_TF_CONVERT_VARIABLES_TO_CONSTANTS"] = "1"
os.environ[
    "TF_ENABLE_ONEDNN_OPTS"] = "1"  # This needs to be set before importing TF
import argparse
import numpy as np
import tensorflow as tf
import openvino_tensorflow as ovtf
import time
import cv2
from PIL import Image
from common.utils import get_input_mode, get_colors, draw_boxes, get_anchors, rename_file
from common.pre_process import preprocess_image_yolov3 as preprocess_image
from common.post_process import yolo3_postprocess_np

# Enable these variables for runtime inference optimizations
os.environ["OPENVINO_TF_CONVERT_VARIABLES_TO_CONSTANTS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"


def load_coco_names(file_name):
    names = {}
    if not os.path.exists(file_name):
        raise AssertionError("could not find label file path")
    with open(file_name) as f:
        for coco_id, name in enumerate(f):
            names[coco_id] = name
    return names


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


if __name__ == "__main__":
    input_file = "examples/data/grace_hopper.jpg"
    model_file = "examples/data/yolo_v4"
    label_file = "examples/data/coco.names"
    anchor_file = "examples/data/yolov4_anchors.txt"
    input_height = 416
    input_width = 416
    backend_name = "CPU"
    output_dir = "."
    conf_threshold = 0.6
    iou_threshold = 0.5

    # overlay parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = .6
    color = (0, 0, 0)
    font_thickness = 2

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", help="Optional. Path to model to be executed.")
    parser.add_argument(
        "--labels", help="Optional. Path to labels mapping file.")
    parser.add_argument(
        "--input",
        help=
        "Optional. The input to be processed. Path to an image or video or directory of images. Use 0 for using camera as input."
    )
    parser.add_argument(
        "--input_height",
        type=int,
        help="Optional. Specify input height value.")
    parser.add_argument(
        "--rename",
        help=
        "Optional. The input image or the directory of the images will be renamed",
        action='store_true')
    parser.add_argument(
        "--input_width", type=int, help="Optional. Specify input width value.")
    parser.add_argument(
        "--backend",
        help="Optional. Specify the target device to infer on; "
        "CPU, GPU, MYRIAD, or VAD-M is acceptable. Default value is CPU.")
    parser.add_argument(
        "--no_show", help="Optional. Don't show output.", action='store_true')
    parser.add_argument(
        "--conf_threshold",
        type=float,
        help="Optional. Specify confidence threshold. Default is 0.6.")
    parser.add_argument(
        "--iou_threshold",
        type=float,
        help="Optional. Specify iou threshold. Default is 0.5.")
    parser.add_argument(
        "--disable_ovtf",
        help="Optional."
        "Disable openvino_tensorflow pass and run on stock TF.",
        action='store_true')

    args = parser.parse_args()
    if args.model:
        model_file = args.model
        if args.labels:
            label_file = args.labels
        else:
            label_file = None
    if args.input:
        input_file = args.input
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.backend:
        backend_name = args.backend
    if args.conf_threshold:
        conf_threshold = args.conf_threshold
    if args.iou_threshold:
        iou_threshold = args.iou_threshold
    if args.rename:
        print(40 * '-')
        print(
            "--rename argument is enabled, this will rename the input image or directory of images in your disk.\n Press 'y' to continue.\n Press 'a' to abort.\n Press any other key to proceed without renaming."
        )
        print(40 * '-')
        val = input()
        if val == 'y':
            print(" Renaming has been enabled")
        elif val == 'a':
            print("Aborted")
            exit(0)
        else:
            print("Renaming has been disabled")
            args.rename = False

    # Load model and process input image
    model = tf.saved_model.load(model_file)

    # Load the labels
    if label_file:
        classes = load_coco_names(label_file)
    anchors = get_anchors(anchor_file)

    if not args.disable_ovtf:
        # Print list of available backends
        print('Available Backends:')
        backends_list = ovtf.list_backends()
        for backend in backends_list:
            print(backend)
        ovtf.set_backend(backend_name)
    else:
        ovtf.disable()

    cap = None
    images = []
    if label_file:
        labels = load_labels(label_file)
    colors = get_colors(labels)
    input_mode = get_input_mode(input_file)
    if input_mode == "video":
        cap = cv2.VideoCapture(input_file)
    elif input_mode == "camera":
        cap = cv2.VideoCapture(0)
    elif input_mode == 'image':
        images = [input_file]
    elif input_mode == 'directory':
        if not os.path.isdir(input_file):
            raise AssertionError("Path doesn't exist {0}".format(input_file))
        images = [
            os.path.join(input_file, fname)
            for fname in os.listdir(input_file)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff',
                                       '.bmp'))
        ]
        result_dir = os.path.join(input_file, '../detections')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
    else:
        raise Exception(
            "Invalid input. Path to an image or video or directory of images. Use 0 for using camera as input."
        )
    images_len = len(images)
    image_id = -1
    # Run inference
    while True:
        image_id += 1
        if input_mode in ['camera', 'video']:
            if cap.isOpened():
                ret, frame = cap.read()
                if ret is True:
                    pass
                else:
                    break
            else:
                break
        if input_mode in ['image', 'directory']:
            if image_id < images_len:
                frame = cv2.imread(images[image_id])
            else:
                break
        img = frame
        image = Image.fromarray(img)
        img_resized = tf.convert_to_tensor(
            preprocess_image(image, (input_height, input_width)))

        # Warmup
        if image_id == 0:
            detected_boxes = model(img_resized)

        # Run
        start = time.time()
        detected_boxes = model(img_resized)
        elapsed = time.time() - start
        fps = 1 / elapsed
        print('Inference time in ms: %.2f' % (elapsed * 1000))
        # post-processing
        image_shape = tuple((frame.shape[0], frame.shape[1]))
        out_boxes, out_classes, out_scores = yolo3_postprocess_np(
            detected_boxes,
            image_shape,
            anchors,
            len(labels), (input_height, input_width),
            max_boxes=10,
            confidence=conf_threshold,
            iou_threshold=iou_threshold,
            elim_grid_sense=True)
        # modified draw_boxes function to return an openCV formatted image
        img_bbox = draw_boxes(img, out_boxes, out_classes, out_scores, labels,
                              colors)
        # draw information overlay onto the frames
        cv2.putText(img_bbox, 'Inference Running on : {0}'.format(backend_name),
                    (30, 50), font, font_size, color, font_thickness)
        cv2.putText(
            img_bbox, 'FPS : {0} | Inference Time : {1}ms'.format(
                int(fps), round((elapsed * 1000), 2)), (30, 80), font,
            font_size, color, font_thickness)
        if input_mode in 'directory':
            out_file = "detections_{0}.jpg".format(image_id)
            out_file = os.path.join(result_dir, out_file)
            cv2.imwrite(out_file, img_bbox)
        if input_mode in 'image':
            cv2.imwrite("detections.jpg", img_bbox)
            print("Output image is saved in detections.jpg")
        if not args.no_show:
            cv2.imshow("detections", img_bbox)
            if cv2.waitKey(1) & 0XFF == ord('q'):
                break
        if args.rename:
            rename_file(images[image_id], out_classes, labels)
    if input_mode in 'directory':
        print("Output images is saved in {0}".format(
            os.path.abspath(result_dir)))
    if cap:
        cap.release()
        cv2.destroyAllWindows()
