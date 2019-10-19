import tensorflow as tf
import ngraph_bridge

def new_device():
  g = tf.Graph()
  with g.as_default():
    inp = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1], name="in")
    with g.device("/device:XPU:0"):
      add = inp + inp
    outp = tf.identity(add, name="out")
    with tf.Session() as sess:
      print(sess.run(outp, feed_dict={inp: [[[1.0]]]}))

new_device()
