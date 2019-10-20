# The following filtering will supress all the `FutureWarning` from numpy
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorflow as tf
import os
os.environ['NGTF_USE_DEVICE'] = "1"
import ngraph_bridge
import sys
import os
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def new_device():
  g = tf.Graph()
  with g.as_default():
    inp = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1], name="in")
    with g.device("/device:NGRAPH:0"):
      add = inp + inp
    with g.device("/device:CPU:0"):    
      mul = add * inp
      outp = tf.identity(mul, name="out")

    config = tf.ConfigProto()
    config.allow_soft_placement = False

    with tf.Session(config=config) as sess:
      print(sess.run(outp, feed_dict={inp: [[[1.0]]]}))

new_device()
