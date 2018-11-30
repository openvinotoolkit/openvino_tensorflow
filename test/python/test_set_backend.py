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

from __future__ import print_function
import pytest
import tensorflow as tf
import ngraph_bridge


# Test ngraph_bridge config options
def test_set_backend():
    ngraph_bridge.enable()
    backend_cpu = 'CPU'
    backend_interpreter = 'INTERPRETER'

    found_cpu = False
    found_interpreter = False
    # These will only print when running pytest with flag "-s"
    print("Number of supported backends ", ngraph_bridge.backends_len())
    supported_backends = ngraph_bridge.list_backends()
    print(" ****** Supported Backends ****** ")
    for backend_name in supported_backends:
        print(backend_name)
        if backend_name == backend_cpu:
            found_cpu = True
        if backend_name == backend_interpreter:
            found_interpreter = True
    print(" ******************************** ")
    assert (found_cpu and found_interpreter) == True

    # Create Graph
    val = tf.placeholder(tf.float32)
    out1 = tf.abs(val)
    out2 = tf.abs(out1)

    # set INTERPRETER backend
    assert ngraph_bridge.is_supported_backend(backend_interpreter) == True
    ngraph_bridge.set_backend(backend_interpreter)
    currently_set_backend = ngraph_bridge.get_currently_set_backend_name()
    assert currently_set_backend == backend_interpreter

    # create new session to execute graph
    # If you want to re-confirm which backend the graph was executed
    # currently the only way is to enable NGRAPH_TF_VLOG_LEVEL=5
    with tf.Session() as sess:
        sess.run((out2,), feed_dict={val: ((1.4, -0.5, -1))})
    currently_set_backend = ngraph_bridge.get_currently_set_backend_name()
    assert currently_set_backend == backend_interpreter

    # set CPU backend
    assert ngraph_bridge.is_supported_backend(backend_cpu) == True
    ngraph_bridge.set_backend(backend_cpu)
    currently_set_backend = ngraph_bridge.get_currently_set_backend_name()
    assert currently_set_backend == backend_cpu
    # create new session to execute graph
    with tf.Session() as sess:
        sess.run((out2,), feed_dict={val: ((1.4, -0.5, -1))})
    currently_set_backend = ngraph_bridge.get_currently_set_backend_name()
    assert currently_set_backend == backend_cpu
