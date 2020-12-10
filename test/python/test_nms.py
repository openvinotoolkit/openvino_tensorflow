# ==============================================================================
#  Copyright 2018-2020 Intel Corporation
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
"""nGraph TensorFlow bridge NMSV2 operation test

"""

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import pytest

from common import NgraphTest


class TestNMSOperations(NgraphTest):

    def test_NMSV2(self):

        boxes = tf.compat.v1.placeholder(tf.float32, shape=(6, 4))
        scores = tf.compat.v1.placeholder(tf.float32, shape=(6))
        max_output_size = tf.compat.v1.placeholder(tf.int32, shape=(None))

        boxes_np = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                    [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
        scores_np = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]
        max_output_size_np = 3

        nmsv2 = tf.raw_ops.NonMaxSuppressionV2(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=0.5)

        def run_test(sess):
            return sess.run(
                (nmsv2,),
                feed_dict={
                    boxes: boxes_np,
                    scores: scores_np,
                    max_output_size: max_output_size_np
                })

        assert np.allclose(
            self.without_ngraph(run_test), self.with_ngraph(run_test))
