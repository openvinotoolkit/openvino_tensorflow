# ==============================================================================
# Copyright (C) 2021 Intel Corporation

# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow installation test

"""
from __future__ import print_function

import tensorflow as tf
import openvino_tensorflow

if __name__ == '__main__':
    print("TensorFlow version: ", tf.version.GIT_VERSION, tf.version.VERSION)
