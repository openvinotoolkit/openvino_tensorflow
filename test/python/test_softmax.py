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
"""nGraph TensorFlow relu6 test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from common import NgraphTest
import tensorflow as tf
import numpy as np

class TestSoftmax(NgraphTest):
    log_placement = False

    def test_softmax_2D(self):

        x = tf.placeholder(tf.float32, shape=(2, 3))
        config = tf.ConfigProto(
            allow_soft_placement=False)


        # input value and expected value
        x_np = np.random.rand(2, 3)
        one_only_on_dim = list(x_np.shape) 
        dim  = len(x_np.shape)-1 
        one_only_on_dim[dim] = 1 
        y_np = np.exp(x_np)
        a_np = y_np/np.reshape(np.sum(y_np,dim),one_only_on_dim)
        expected = a_np
        with tf.device("/device:NGRAPH:0"): 
            a = tf.nn.softmax(x) 
            with tf.Session(config=config) as sess:
                print("Python: Running with Session")
                (result_a) = sess.run(
                    (a),
                    feed_dict={
                        x: x_np,
                    })
        print("result:", result_a)
        print("expected:", expected)
        atol = 1e-5
        error = np.absolute(result_a-expected)
        assert np.amax(error) <= atol
    
    

                
