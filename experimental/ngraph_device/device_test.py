# The following filtering will supress all the `FutureWarning` from numpy
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import os
os.environ['NGRAPH_TF_USE_DEVICE_MODE'] = "1"
import ngraph_bridge
import sys
import os

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def new_device():
    g = tf.Graph()
    with g.as_default():
        inp = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[None, 1, 1], name="in")
        with g.device("/device:NGRAPH:0"):
            add = inp + 10
        with g.device("/device:CPU:0"):
            mul = add * inp
            outp = tf.identity(mul, name="out")

        config = tf.compat.v1.ConfigProto(
            allow_soft_placement=False,
            log_device_placement=False,
            inter_op_parallelism_threads=1,
            graph_options=tf.compat.v1.GraphOptions(
                optimizer_options=tf.compat.v1.OptimizerOptions(
                    opt_level=tf.compat.v1.OptimizerOptions.L0,
                    do_common_subexpression_elimination=False,
                    do_constant_folding=False,
                    do_function_inlining=False,
                )))

        with tf.compat.v1.Session(config=config) as sess:
            print(sess.run(outp, feed_dict={inp: [[[1.0]]]}))


new_device()
