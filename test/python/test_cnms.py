# ==============================================================================
#  Copyright 2019-2020 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
"""nGraph TensorFlow bridge floor operation test
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
from common import NgraphTest
from google.protobuf import text_format
from tensorflow.python.ops.gen_image_ops import combined_non_max_suppression


class TestFloorOperations(NgraphTest):

    @pytest.mark.skip(reason="Backend specific test")
    def test_cmns(self):
        input_boxes = np.array(
            [[[0, 0, 1, 1], [0, 0, 4, 5]], [[0, 0.1, 1, 1.1], [0, 0.1, 2, 1.1]],
             [[0, -0.1, 1, 0.9], [0, -0.1, 1, 0.9]],
             [[0, 10, 1, 11], [0, 10, 1, 11]],
             [[0, 10.1, 1, 11.1], [0, 10.1, 1, 11.1]],
             [[0, 100, 1, 101], [0, 100, 1, 101]],
             [[0, 1000, 1, 1002], [0, 999, 2, 1004]],
             [[0, 1000, 1, 1002.1], [0, 999, 2, 1002.7]]], np.float32)
        input_scores = np.array([[.9, 0.01], [.75, 0.05], [.6, 0.01], [.95, 0],
                                 [.5, 0.01], [.3, 0.01], [.01, .85], [.01, .5]],
                                np.float32)

        input_boxes = np.reshape(input_boxes, (1, 8, 2, 4))
        input_scores = np.reshape(input_scores, (1, 8, 2))
        score_thresh = 0.1
        iou_thresh = .5
        max_size_per_class = 4
        max_output_size = 5

        boxes_ph = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=(1, 8, 2, 4), name='box_input')
        scores_ph = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=(1, 8, 2), name='scores_input')
        cnms = combined_non_max_suppression(
            boxes_ph,
            scores_ph,
            max_size_per_class,
            max_output_size,
            iou_thresh,
            score_thresh,
            pad_per_class=False,
            clip_boxes=True)

        def run_test(sess):
            return sess.run(
                cnms,
                feed_dict={
                    boxes_ph: input_boxes,
                    scores_ph: input_scores
                })

        ngb_out = self.with_ngraph(run_test)
        tf_out = self.without_ngraph(run_test)
        assert len(ngb_out) == len(tf_out)
        for res1, res2 in zip(ngb_out, tf_out):
            assert res1.shape == res2.shape
            assert np.isclose(res1, res2).all()
