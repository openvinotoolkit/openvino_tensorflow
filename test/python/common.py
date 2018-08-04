import platform

import tensorflow as tf
import random

__all__ = ['LIBNGRAPH_DEVICE', 'NgraphTest']


_ext = 'dylib' if platform.system() == 'Darwin' else 'so'

LIBNGRAPH_DEVICE = 'libngraph_device.' + _ext


class NgraphTest(object):
  test_device = "/device:NGRAPH:0"
  cpu_test_device = "/device:CPU:0"
  soft_placement = False
  log_placement = False

  @property
  def device(self):
    return tf.device(self.test_device)

  @property
  def cpu_device(self):
    return tf.device(self.cpu_test_device)

  @property
  def session(self):
    return tf.Session(config=self.config)

  @property
  def config(self):
    return tf.ConfigProto(
        allow_soft_placement=self.soft_placement,
        log_device_placement=self.log_placement,
        inter_op_parallelism_threads=1)
  
  # returns a vector of length 'vector_length' with random
  # float numbers in range [start,end]
  def generate_random_numbers(self, vector_length, start, end):
    return [random.uniform(start, end) for i in range(vector_length)]
