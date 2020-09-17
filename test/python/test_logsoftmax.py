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
"""nGraph TensorFlow bridge split operation test

"""

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import pytest

from common import NgraphTest


class TestLogSoftmaxOperations(NgraphTest):

    def test_logsoftmax(self):
        type = np.float32
        max = np.finfo(type).max
        features = np.array([[1., 1., 1., 1.], [max, 1., 2., 3.]]).astype(type)
        logsoftmax = tf.nn.log_softmax(features)
        sess_fn = lambda sess: sess.run([logsoftmax])
        out = self.with_ngraph(sess_fn)
        assert np.allclose(
            np.array([[-1.386294, -1.386294, -1.386294, -1.386294],
                      [0, -max, -max, -max]]),
            out,
            rtol=1.e-5,
            atol=1.e-5)
