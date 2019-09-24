import tensorflow as tf
import ngraph_bridge
from google.protobuf import text_format
import pdb


def get_graphdef(pb_filename):
    graph_def = tf.GraphDef()
    if pb_filename.endswith("pbtxt"):
        with open(pb_filename, "r") as f:
            text_format.Merge(f.read(), graph_def)
    else:
        with open(pb_filename, "rb") as f:
            graph_def.ParseFromString(f.read())
    return graph_def


inputs = ['import/x:0', 'import/y:0']
with tf.Graph().as_default() as graph:
    tf.import_graph_def(get_graphdef('axpy_ngraph.pbtxt'))
config = tf.ConfigProto(
    inter_op_parallelism_threads=1, allow_soft_placement=True)
sess = tf.Session(graph=graph, config=config)

xval = [[1, 1, 1], [2, 2, 2]]
yval = xval
out = sess.run(
    graph.get_tensor_by_name('import/add:0'),
    feed_dict={
        graph.get_tensor_by_name('import/x:0'): xval,
        graph.get_tensor_by_name('import/y:0'): yval
    })
print(out)
