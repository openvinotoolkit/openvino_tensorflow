# ==============================================================================
#  Copyright 2018 Intel Corporation
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
"""nGraph TensorFlow L2loss test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pytest

from common import NgraphTest


class TestL2Loss(NgraphTest):

    @pytest.mark.parametrize(("xshape"), ((3, 4, 5), (1,)))
    def test_l2loss(self, xshape):
        x = tf.placeholder(tf.float32, shape=xshape)
        out = tf.nn.l2_loss(x)
        values = np.random.rand(*xshape)
        sess_fn = lambda sess: sess.run((out), feed_dict={x: values})
        assert np.allclose(
            self.with_ngraph(sess_fn), self.without_ngraph(sess_fn))

    def test_l2loss_empty(self):
        x = tf.placeholder(tf.float32, shape=())
        out = tf.nn.l2_loss(x)
        sess_fn = lambda sess: sess.run((out), feed_dict={x: None})

        # expect to be nan
        assert (self.with_ngraph(sess_fn) != self.without_ngraph(sess_fn))
