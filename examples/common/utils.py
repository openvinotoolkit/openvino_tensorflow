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
# Modified from the repository (https://github.com/david8862/keras-YOLOv3-model-set):
# https://github.com/david8862/keras-YOLOv3-model-set/blob/master/common/utils.py

import os
import numpy as np
import time
import cv2, colorsys
from PIL import Image
import imghdr
import tensorflow as tf
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def get_input_mode(input_path):
    if str(input_path).lower() == '0':
        return "camera"
    assert os.path.exists(input_path), "input path doesn't exist"
    if os.path.isdir(input_path):
        images = os.listdir(input_path)
        if len(images) < 1:
            assert False, "Input directory doesn't contain any images"
        for i in images:
            image_path = os.path.join(input_path, i)
            if imghdr.what(image_path) == None:
                assert False, "Input directory contains non image files"
        return "directory"
    elif os.path.isfile(input_path):
        if imghdr.what(input_path) != None:
            return "image"
        elif input_path.rsplit('.', 1)[1] in ['mp4', 'avi']:
            return "video"
    else:
        return "Invalid input"


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()
    assert os.path.exists(model_file), "Could not find model path"
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def get_anchors():
    anchors = [
        116, 90, 156, 198, 373, 326, 30, 61, 62, 45, 59, 119, 10, 13, 16, 30,
        33, 23
    ]
    anchors = [float(x) for x in anchors]
    return np.array(anchors).reshape(-1, 2)


def get_colors(class_names):
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [
        (x / len(class_names), 1., 1.) for x in range(len(class_names))
    ]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.
    return colors


def draw_label(image, text, color, coords):
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.
    (text_width, text_height) = cv2.getTextSize(
        text, font, fontScale=font_scale, thickness=1)[0]

    padding = 5
    rect_height = text_height + padding * 2
    rect_width = text_width + padding * 2

    (x, y) = coords

    cv2.rectangle(image, (x, y), (x + rect_width, y - rect_height), color,
                  cv2.FILLED)
    cv2.putText(
        image,
        text, (x + padding, y - text_height + padding),
        font,
        fontScale=font_scale,
        color=(255, 255, 255),
        lineType=cv2.LINE_AA)

    return image


def draw_boxes(image,
               boxes,
               classes,
               scores,
               class_names,
               colors,
               show_score=True):
    if classes is None or len(classes) == 0:
        return image

    for box, cls, score in zip(boxes, classes, scores):
        xmin, ymin, xmax, ymax = box

        class_name = class_names[cls]
        if show_score:
            label = '{} {:.2f}'.format(class_name, score)
        else:
            label = '{}'.format(class_name)

        # if no color info, use black(0,0,0)
        if colors == None:
            color = (0, 0, 0)
        else:
            color = colors[cls]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1, cv2.LINE_AA)
        image = draw_label(image, label, color, (xmin, ymin))

    return image
