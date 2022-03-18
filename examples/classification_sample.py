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

# Modified from TensorFlow example:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/label_image.py
#https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/image_classification.ipynb
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
# Enable these variables for runtime inference optimizations
os.environ["OPENVINO_TF_CONVERT_VARIABLES_TO_CONSTANTS"] = "1"
os.environ[
    "TF_ENABLE_ONEDNN_OPTS"] = "1"  # This needs to be set before importing TF
import numpy as np
import tensorflow as tf
import openvino_tensorflow as ovtf
import tensorflow_hub as hub
from PIL import Image
import time
import cv2

from common.utils import get_input_mode

# Enable these variables for runtime inference optimizations
os.environ["OPENVINO_TF_CONVERT_VARIABLES_TO_CONSTANTS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"


def preprocess_image(frame,
                     input_height=299,
                     input_width=299,
                     input_mean=0,
                     input_std=255):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    resized_image = image.resize((input_height, input_width))
    resized_image = np.asarray(resized_image, np.float32)
    normalized_image = (resized_image - input_mean) / input_std
    result = np.expand_dims(normalized_image, 0)
    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


if __name__ == "__main__":
    input_file = tf.keras.utils.get_file(
        'grace_hopper.jpg',
        "https://www.tensorflow.org/images/grace_hopper.jpg")
    model_file = ""
    label_file = tf.keras.utils.get_file(
        'ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    )
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    backend_name = "CPU"

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
        "Optional. The input to be processed. Path to an image or video or directory of images. Use 0 for using camera as input"
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
        "CPU, GPU, MYRIAD or VAD-M is acceptable. Default value is CPU.")
    parser.add_argument(
        "--no_show", help="Optional. Don't show output.", action='store_true')
    parser.add_argument(
        "--disable_ovtf",
        help="Optional. Disable openvino_tensorflow pass and run on stock TF.",
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
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.backend:
        backend_name = args.backend

    if model_file == "":
        model = hub.load(
            "https://tfhub.dev/google/imagenet/inception_v3/classification/4")
    else:
        model = tf.saved_model.load(model_file)

    if not args.disable_ovtf:
        #Print list of available backends
        print('Available Backends:')
        backends_list = ovtf.list_backends()
        for backend in backends_list:
            print(backend)
        ovtf.set_backend(backend_name)
    else:
        ovtf.disable()

    #Load the labels
    cap = None
    images = []
    if label_file:
        labels = load_labels(label_file)
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
    else:
        raise Exception(
            "Invalid input. Path to an image or video or directory of images. Use 0 for using camera as input."
        )
    images_len = len(images)
    # Initialize session and run
    image_id = -1
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

        t = tf.convert_to_tensor(
            preprocess_image(
                frame, input_height=input_height, input_width=input_width))

        # Warmup
        if image_id == 0:
            results = model(t)

        # run
        start = time.time()
        results = model(t)
        elapsed = time.time() - start
        fps = 1 / elapsed
        print('Inference time in ms: %.2f' % (elapsed * 1000))

        results = tf.nn.softmax(results).numpy()
        if label_file:
            cv2.putText(frame,
                        'Inference Running on : {0}'.format(backend_name),
                        (30, 50), font, font_size, color, font_thickness)
            cv2.putText(
                frame, 'FPS : {0} | Inference Time : {1}ms'.format(
                    int(fps), round((elapsed * 1000), 2)), (30, 80), font,
                font_size, color, font_thickness)
            top_5 = tf.argsort(
                results, axis=-1, direction="DESCENDING")[0][:5].numpy()
            c = 130
            for i, item in enumerate(top_5):
                cv2.putText(
                    frame, '{0} : {1}'.format(labels[item],
                                              results[0][top_5][i]), (30, c),
                    font, font_size, color, font_thickness)
                print(labels[item], results[0][top_5][i])
                c += 30
        else:
            print("No label file provided. Cannot print classification results")
        if not args.no_show:
            cv2.imshow("results", frame)
            if cv2.waitKey(1) & 0XFF == ord('q'):
                break
    if cap:
        cap.release()
    cv2.destroyAllWindows()
