import platform

import tensorflow as tf


__all__ = ['LIBNGRAPH_DEVICE', 'NgraphTest']


_ext = 'dylib' if platform.system() == 'Darwin' else 'so'

LIBNGRAPH_DEVICE = 'libngraph_device.' + _ext


class NgraphTest(object):
  test_device = "/device:NGRAPH:0"
  soft_placement = False
  log_placement = False

  @property
  def device(self):
    return tf.device(self.test_device)

  @property
  def session(self):
    return tf.Session(config=self.config)

  @property
  def config(self):
    return tf.ConfigProto(
        allow_soft_placement=self.soft_placement,
        log_device_placement=self.log_placement,
        inter_op_parallelism_threads=1)
