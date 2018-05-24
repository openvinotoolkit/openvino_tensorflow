import sys 

import ngraph
import numpy as np
import tensorflow as tf

a = np.full((2, 3), 5.0)
x = tf.placeholder(tf.float32, [None, 3], name='x')
y = tf.placeholder(tf.float32, shape=(2, 3), name='y')

with tf.device("/device:NGRAPH:0"): 
   c = a * x
   axpy = c + y
