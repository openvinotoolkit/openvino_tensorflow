import os
import platform
import random

import tensorflow as tf


__all__ = ['LIBNGRAPH_BRIDGE', 'NgraphTest']


_ext = 'dylib' if platform.system() == 'Darwin' else 'so'

LIBNGRAPH_BRIDGE = 'libngraph_bridge.' + _ext


class NgraphTest(object):
  def with_ngraph(self, l, config=tf.ConfigProto()):
    ngraph_tf_disable = os.environ.pop('NGRAPH_TF_DISABLE', None)
    ngraph_tf_disable_deassign_clusters = os.environ.pop(
        'NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS', None)

    os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = '1'

    with tf.Session(config=config) as sess:
      retval = l(sess)

    os.environ.pop('NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS', None)

    if ngraph_tf_disable is not None:
      os.environ['NGRAPH_TF_DISABLE'] = ngraph_tf_disable
    if ngraph_tf_disable_deassign_clusters is not None:
      os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = \
          ngraph_tf_disable_deassign_clusters

    return retval

  def without_ngraph(self, l, config=tf.ConfigProto()):
    ngraph_tf_disable = os.environ.pop('NGRAPH_TF_DISABLE', None)
    ngraph_tf_disable_deassign_clusters = os.environ.pop(
        'NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS', None)

    os.environ['NGRAPH_TF_DISABLE'] = '1'

    with tf.Session(config=config) as sess:
      retval = l(sess)

    os.environ.pop('NGRAPH_TF_DISABLE', None)

    if ngraph_tf_disable is not None:
      os.environ['NGRAPH_TF_DISABLE'] = ngraph_tf_disable
    if ngraph_tf_disable_deassign_clusters is not None:
      os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = \
          ngraph_tf_disable_deassign_clusters

    return retval

  # returns a vector of length 'vector_length' with random
  # float numbers in range [start,end]
  def generate_random_numbers(self, vector_length, start, end,
                              datatype="DTYPE_FLOAT"):
    if datatype == "DTYPE_INT":
      return [random.randint(start, end) for i in range(vector_length)]
    return [random.uniform(start, end) for i in range(vector_length)]
