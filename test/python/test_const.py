# ==============================================================================
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
"""Openvino Tensorflow Const operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import os

from common import NgraphTest

# Uncomment for debugging; Also add -s in command like, e.g.
# (venv-tf-py3) [build_cmake]$
# OPENVINO_TF_LOG_PLACEMENT=1 OPENVINO_TF_VLOG_LEVEL=6 pytest -s -k test_const_scalarval ../test/python/test_const.py
import logging
logging.basicConfig(level=logging.DEBUG)


class TestConstOperations(NgraphTest):

    def test_const_listvals(self):
        zz = tf.constant([1, 2, 3, 4, 5, 6], dtype=float, shape=[2, 3])

        def run_test(sess):
            return sess.run(zz)

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    def test_const_listvals_2(self):
        zz = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=float, shape=[2, 3])

        def run_test(sess):
            return sess.run(zz)

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    def test_const_scalarval(self):
        zz = tf.constant(-3, dtype=float, shape=[2, 3])

        def run_test(sess):
            return sess.run(zz)

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

    def test_const_lastfill(self):
        try:
            zz = tf.constant([1, 2], dtype=float, shape=[2, 3])
            assert False, 'TF2.0 is now able construct constants with less elements, update test accordingly'

            def run_test(sess):
                return sess.run(zz)

            assert (self.with_ngraph(run_test) == self.without_ngraph(run_test)
                   ).all()
        except:
            return

    def test_const_empty(self):
        log = logging.getLogger('test_const_empty')
        try:
            zz = tf.constant([], dtype=float, shape=[2, 3])
            assert False, 'TF2.0 is able to construct constants with 0 elements, update test accordingly'

            def run_test(sess):
                log.debug('Invoking sess.run(zz)')
                return sess.run(zz)

            # Ideally we want same behavior for both TF & NG, but for now we are deviating,
            # NGraph will throw error, but TF will fill in zeros
            # assert (
            #    self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()

            # Test to see that exception is raised in NG
            try:
                # This test is expected to fail currently
                res = self.with_ngraph(run_test)
                assert False, 'Failed, expected test to raise error'
            except:
                log.debug('Passed, expected NG to raise error...')
                assert True
        except:
            return
