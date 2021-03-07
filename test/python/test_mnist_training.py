# ==============================================================================
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================

import sys
import pytest
import getpass
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import openvino_tensorflow

import numpy as np
from common import NgraphTest

# This test needs access to mnist_deep_simplified.py script
# present in openvino_tensorflow/examples/mnist
# so this path needs to be added to python path when running this

# For eg. when running the test from openvino_tensorflow/build_cmake/test/python
# you can add this path as below
sys.path.insert(0, '../../../examples/mnist')

from mnist_deep_simplified import *


class TestMnistTraining(NgraphTest):

    @pytest.mark.parametrize(("optimizer"), ("adam", "sgd", "momentum"))
    def test_mnist_training(self, optimizer):

        class mnist_training_flags:

            def __init__(self, data_dir, model_dir, training_iterations,
                         training_batch_size, validation_batch_size,
                         make_deterministic, training_optimizer):
                self.data_dir = data_dir
                self.model_dir = model_dir
                self.train_loop_count = training_iterations
                self.batch_size = training_batch_size
                self.test_image_count = validation_batch_size
                self.make_deterministic = make_deterministic
                self.optimizer = optimizer

        data_dir = '/tmp/' + getpass.getuser() + 'tensorflow/mnist/input_data'
        train_loop_count = 50
        batch_size = 50
        test_image_count = None
        make_deterministic = True
        model_dir = './mnist_trained/'

        FLAGS = mnist_training_flags(data_dir, model_dir, train_loop_count,
                                     batch_size, test_image_count,
                                     make_deterministic, optimizer)

        # Run on nGraph
        ng_loss_values, ng_test_accuracy = train_mnist_cnn(FLAGS)
        ng_values = ng_loss_values + [ng_test_accuracy]
        # Reset the Graph
        tf.compat.v1.reset_default_graph()

        # disable ngraph-tf
        openvino_tensorflow.disable()
        tf_loss_values, tf_test_accuracy = train_mnist_cnn(FLAGS)
        tf_values = tf_loss_values + [tf_test_accuracy]

        # compare values
        assert np.allclose(
            ng_values, tf_values,
            atol=1e-3), "Loss or Accuracy values don't match"
