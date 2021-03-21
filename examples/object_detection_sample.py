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
# Modified from TensorFlow example:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/label_image.py
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import tensorflow as tf
import openvino_tensorflow
import time
import cv2
from subprocess import check_output, call

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(image_file,
                                input_height=416,
                                input_width=416,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.io.read_file(image_file, input_name)
    if image_file.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif image_file.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif image_file.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.compat.v1.image.resize_bilinear(dims_expander,
                                                 [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.compat.v1.Session()
    result = sess.run(normalized)

    return result

def process_image(image_path, input_height, input_width):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image,(input_width,input_height))
    resized_image = resized_image / 255.0

    return resized_image, image


def draw_boxes(output_filename, classes_filename, inputs, original_image, resized_image):
    names = {}
    with open(classes_filename) as f:
        class_names = f.readlines()
        for id, name in enumerate(class_names):
            names[id] = name
    
    height_ratio = original_image.shape[0] / resized_image.shape[0]
    width_ratio = original_image.shape[1] / resized_image.shape[1]
    ratio = (width_ratio, height_ratio)

    for object_class, box_coords_and_prob in inputs.items():
        for box_coord, object_prob in box_coords_and_prob:

            box_coord = box_coord.reshape(2,2) * ratio
            box_coord = box_coord.reshape(-1)

            x0y0 = (int(box_coord[0]),int(box_coord[1]))
            x1y1 = (int(box_coord[2]), int(box_coord[3]))

            textx0y0 = (x0y0[0],x0y0[1]-4)

            cv2.rectangle(original_image, x0y0, x1y1, (255,255,255), 2)
            text_label = str(names[object_class])[:-1] + ", " + str(round(object_prob*100,2)) + "%"
            cv2.putText(original_image, text_label, textx0y0, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imwrite(output_filename, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))

def non_max_suppression(predictions_with_boxes, confidence_threshold, iou_threshold=0.4):
    
    def iou(box1, box2):
        b1_x0, b1_y0, b1_x1, b1_y1 = box1
        b2_x0, b2_y0, b2_x1, b2_y1 = box2

        int_x0 = max(b1_x0, b2_x0)
        int_y0 = max(b1_y0, b2_y0)
        int_x1 = min(b1_x1, b2_x1)
        int_y1 = min(b1_y1, b2_y1)

        int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

        b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
        b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

        iou = int_area / (b1_area + b2_area - int_area + 1e-05)

        return iou
    
    conf_mask = np.expand_dims((predictions_with_boxes[:, :, 5] > confidence_threshold), -1)
    predictions = predictions_with_boxes * conf_mask
    
    result = {}
    
    for i, image_pred in enumerate(predictions):
        shape = image_pred.shape
        non_zero_idxs = np.nonzero(image_pred)
        image_pred = image_pred[non_zero_idxs]
        image_pred = image_pred.reshape(-1, shape[-1])
    
        t_bbox_attrs = image_pred[:, :6]
        bbox_attrs = np.delete(t_bbox_attrs,4,axis=1)
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)
    
        unique_classes = list(set(classes.reshape(-1)))
        
        for cls in unique_classes:
          cls_mask = classes == cls
          cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
          cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
          cls_scores = cls_boxes[:, -1]
          cls_boxes = cls_boxes[:, :-1]
               
          while len(cls_boxes) > 0:
            box = cls_boxes[0]
            score = cls_scores[0]
            if not cls in result:
              result[cls] = []
            result[cls].append((box, score))
            cls_boxes = cls_boxes[1:]
            cls_scores = cls_scores[1:]
            ious = np.array([iou(box, x) for x in cls_boxes])
            iou_mask = ious < iou_threshold
            cls_boxes = cls_boxes[np.nonzero(iou_mask)]
            cls_scores = cls_scores[np.nonzero(iou_mask)]
    
    return result

def convert_box_coordinates(detections):
    split = np.split(detections, [1, 2, 3, 4, 85], axis=2)
    center_x = split[0]
    center_y = split[1]
    width = split[2]
    height = split[3] 
    attrs = split[4]
    
    w2 = width / 2
    h2 = height / 2
    x0 = center_x - w2
    y0 = center_y - h2
    x1 = center_x + w2
    y1 = center_y + h2
    
    boxes = np.concatenate([x0, y0, x1, y1], axis=-1)
    detections = np.concatenate([boxes, attrs], axis=-1)
    return detections


if __name__ == "__main__":
    image_file = "examples/data/grace_hopper.jpg"
    model_file = "examples/data/frozen_darknet_yolov3_model.pb"
    label_file = "examples/data/coco.names"
    input_height = 416
    input_width = 416
    input_mean = 0
    input_std = 255
    input_layer = "inputs"
    output_layer = "output_boxes"
    backend_name = "CPU"

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--backend", help="backend option. Default is CPU")
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
    if args.image:
      image_file = args.image
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

    graph = load_graph(model_file)
    input_tensor = read_tensor_from_image_file(
        image_file,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)

    image, original_image = process_image(image_file, input_height, input_width)
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    #Print list of available backends
    print('Available Backends:')
    backends_list = openvino_tensorflow.list_backends()
    for backend in backends_list:
      print(backend)
    openvino_tensorflow.set_backend(backend_name)
    
    # update config params for openvino tensorflow addon
    config = tf.compat.v1.ConfigProto()
    config_ngraph_enabled = openvino_tensorflow.update_config(config)

    with tf.compat.v1.Session(
            graph=graph,config=config_ngraph_enabled) as sess:
        # Warmup
        results = sess.run(output_operation.outputs[0],
                           {input_operation.outputs[0]: input_tensor})
        # Run
        import time
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                           {input_operation.outputs[0]: input_tensor})
        elapsed = time.time() - start
        print('Inference time in ms: %f' % (elapsed*1000))
          
    # convert box coordinates, apply nms, and draw boxes
    boxes = convert_box_coordinates(results)
    filtered_boxes = non_max_suppression(boxes, confidence_threshold=0.99,iou_threshold=0.5)
    #print(filtered_boxes)
    draw_boxes("detections.jpg",label_file,filtered_boxes,original_image, image)
        
