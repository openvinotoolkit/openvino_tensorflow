# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
# Modified from tensorflow object detection examples:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/label_image.py
# https://github.com/mystic123/tensorflow-yolo-v3/blob/master/utils.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
import tensorflow as tf
import openvino_tensorflow as ovtf
import time
import cv2
from PIL import Image
from common.utils import get_input_mode, load_graph, get_colors, draw_boxes, get_anchors
from common.pre_process import preprocess_image
from common.post_process import yolo3_postprocess_np


def load_coco_names(file_name):
    names = {}
    assert os.path.exists(file_name), "could not find label file path"
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
    model_file = "examples/data/yolo_v3_darknet_2.pb"
    label_file = "examples/data/coco.names"
    input_height = 416
    input_width = 416
    input_mean = 0
    input_std = 255
    input_layer = "image_input"
    output_layer = [
        "conv2d_58/BiasAdd", "conv2d_66/BiasAdd", "conv2d_74/BiasAdd"
    ]
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
        "--graph", help="Optional. Path to graph/model to be executed.")
    parser.add_argument("--input_layer", help="Optional. Name of input layer.")
    parser.add_argument(
        "--output_layer", help="Optional. Name of output layer.")
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
        "--input_width", type=int, help="Optional. Specify input width value.")
    parser.add_argument(
        "--input_mean", type=int, help="Optional. Specify input mean value.")
    parser.add_argument(
        "--input_std", type=int, help="Optional. Specify input std value.")
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
    if args.graph:
        model_file = args.graph
        if not args.input_layer:
            raise Exception("Specify input layer for this network")
        else:
            input_layer = args.input_layer
        if not args.output_layer:
            raise Exception("Specify output layer for this network")
        else:
            output_layer = args.output_layer
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
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.backend:
        backend_name = args.backend
    if args.conf_threshold:
        conf_threshold = args.conf_threshold
    if args.iou_threshold:
        iou_threshold = args.iou_threshold

    # Load graph and process input image
    graph = load_graph(model_file)
    # Load the labels
    if label_file:
        classes = load_coco_names(label_file)
    input_name = "import/" + input_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = [
        graph.get_operation_by_name("import/" + output_name)
        for output_name in output_layer
    ]

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
        images = [os.path.join(input_file, i) for i in os.listdir(input_file)]
    else:
        raise Exception(
            "Invalid input. Path to an image or video or directory of images. Use 0 for using camera as input."
        )
    images_len = len(images)
    image_id = -1
    # Initialize session and run
    config = tf.compat.v1.ConfigProto()
    with tf.compat.v1.Session(graph=graph, config=config) as sess:
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
            img_resized = preprocess_image(image, (input_height, input_width))
            # Run
            start = time.time()
            detected_boxes = sess.run(
                (output_operation[0].outputs[0], output_operation[1].outputs[0],
                 output_operation[2].outputs[0]),
                {input_operation.outputs[0]: [img_resized]})
            elapsed = time.time() - start
            fps = 1 / elapsed
            print('Inference time in ms: %.2f' % (elapsed * 1000))
            # post-processing
            image_shape = tuple((frame.shape[0], frame.shape[1]))
            out_boxes, out_classes, out_scores = yolo3_postprocess_np(
                detected_boxes,
                image_shape,
                get_anchors(),
                len(labels), (input_height, input_width),
                max_boxes=10,
                elim_grid_sense=True)

            # modified draw_boxes function to return an openCV formatted image
            img_bbox = draw_boxes(img, out_boxes, out_classes, out_scores,
                                  labels, colors)
            # draw information overlay onto the frames
            cv2.putText(img_bbox,
                        'Inference Running on : {0}'.format(backend_name),
                        (30, 50), font, font_size, color, font_thickness)
            cv2.putText(
                img_bbox, 'FPS : {0} | Inference Time : {1}ms'.format(
                    int(fps), round((elapsed * 1000), 2)), (30, 80), font,
                font_size, color, font_thickness)
            if not args.no_show:
                cv2.imshow("detections", img_bbox)
                if cv2.waitKey(1) & 0XFF == ord('q'):
                    break
    sess.close()
    if cap:
        cap.release()
    cv2.destroyAllWindows()
