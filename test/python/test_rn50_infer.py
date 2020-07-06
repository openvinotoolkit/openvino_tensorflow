#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Get pretrained model file: wget https://zenodo.org/record/2535873/files/resnet50_v1.pb

import time
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes
import ngraph_bridge

INPUTS = 'input_tensor'
OUTPUTS = 'softmax_tensor'

RESNET_IMAGE_SIZE = 224


class rn50_graph:
    """Evaluate image classifier with optimized TensorFlow graph"""

    def __init__(self):
        arg_parser = ArgumentParser(description='Parse arguments')
        arg_parser.add_argument(
            "--batch-size", dest="batch_size", type=int, default=8)
        arg_parser.add_argument(
            "--num-images", dest='num_images', type=int, default=500)
        arg_parser.add_argument(
            "--num-inter-threads",
            dest='num_inter_threads',
            type=int,
            default=0)
        arg_parser.add_argument(
            "--num-intra-threads",
            dest='num_intra_threads',
            type=int,
            default=0)
        arg_parser.add_argument(
            "--input-graph",
            dest='input_graph',
            type=str,
            default="resnet50_v1.pb")
        self.args = arg_parser.parse_args()

    def run(self):
        """run benchmark with optimized graph"""

        print("Run inference with dummy data")

        config = tf.compat.v1.ConfigProto()
        config.intra_op_parallelism_threads = self.args.num_intra_threads
        config.inter_op_parallelism_threads = self.args.num_inter_threads
        config.use_per_session_threads = 1

        data_graph = tf.Graph()
        with data_graph.as_default():
            input_shape = [
                self.args.batch_size, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3
            ]
            images = tf.random.uniform(
                input_shape,
                0.0,
                255.0,
                dtype=tf.float32,
                seed=42,
                name='synthetic_images')

        infer_graph = tf.Graph()
        with infer_graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.args.input_graph, 'rb') as input_file:
                input_graph_content = input_file.read()
                graph_def.ParseFromString(input_graph_content)
            output_graph = optimize_for_inference(
                graph_def, [INPUTS], [OUTPUTS], dtypes.float32.as_datatype_enum,
                False)
            tf.import_graph_def(output_graph, name='')

        input_tensor = infer_graph.get_tensor_by_name('input_tensor:0')
        output_tensor = infer_graph.get_tensor_by_name('softmax_tensor:0')

        data_sess = tf.compat.v1.Session(graph=data_graph, config=config)
        infer_sess = tf.compat.v1.Session(graph=infer_graph, config=config)

        num_processed_images = 0
        num_remaining_images = self.args.num_images
        total_accuracy1, total_accuracy5 = (0.0, 0.0)

        ngraph_bridge.disable()
        while num_remaining_images >= self.args.batch_size:
            np_images = data_sess.run(images)
            num_processed_images += self.args.batch_size
            num_remaining_images -= self.args.batch_size

            tf_start_time = time.time()
            tf_predictions = infer_sess.run(output_tensor,
                                            {input_tensor: np_images})
            tf_elapsed_time = time.time() - tf_start_time

            np_labels = np.argmax(tf_predictions, axis=-1)

            ngraph_bridge.enable()
            ngtf_start_time = time.time()
            ngtf_predictions = infer_sess.run(output_tensor,
                                              {input_tensor: np_images})
            ngtf_elapsed_time = time.time() - ngtf_start_time
            ngraph_bridge.disable()

            with tf.Graph().as_default():
                accuracy1 = tf.reduce_sum(
                    input_tensor=tf.cast(
                        tf.nn.in_top_k(
                            predictions=tf.constant(ngtf_predictions),
                            targets=tf.constant(np_labels),
                            k=1), tf.float32))

                accuracy5 = tf.reduce_sum(
                    input_tensor=tf.cast(
                        tf.nn.in_top_k(
                            predictions=tf.constant(ngtf_predictions),
                            targets=tf.constant(np_labels),
                            k=5), tf.float32))

                with tf.compat.v1.Session() as accu_sess:
                    np_accuracy1, np_accuracy5 = accu_sess.run(
                        [accuracy1, accuracy5])

                total_accuracy1 += np_accuracy1
                total_accuracy5 += np_accuracy5

            print("Iteration time (TF): %0.4f ms" % tf_elapsed_time)
            print("Iteration time (NGTF): %0.4f ms" % ngtf_elapsed_time)
            print("Processed %d images. (Top1 accuracy, Top5 accuracy) = (%0.4f, %0.4f)" \
                      % (num_processed_images, total_accuracy1 / num_processed_images,
                          total_accuracy5 / num_processed_images))
            assert (total_accuracy1 >= 0.99999 and total_accuracy5 >= 0.99999)


if __name__ == "__main__":
    graph = rn50_graph()
    graph.run()
