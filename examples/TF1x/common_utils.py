import os
import imghdr
import tensorflow as tf


def get_input_mode(input_path):
    if input_path.lower() in ['cam', 'camera']:
        return "camera"
    assert os.path.exists(input_path), "input path doesn't exist"
    if os.path.isdir(input_path):
        images = os.listdir(input_path)
        if len(images) < 1:
            assert False, "Input directory doesn't contain any images"
        for i in images:
            image_path = os.path.join(input_path, i)
            if imghdr.what(image_path) == None:
                assert False, "Input directory contains non image files"
        return "directory"
    elif os.path.isfile(input_path):
        if imghdr.what(input_path) != None:
            return "image"
        elif input_path.rsplit('.', 1)[1] in ['mp4', 'avi']:
            return "video"


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()
    assert os.path.exists(model_file), "Could not find model path"
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph
