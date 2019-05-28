# ==============================================================================
#  Copyright 2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================

import argparse
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.grappler import tf_optimizer
import ngraph_bridge
import os
import sys
from functools import partial


def run_ngraph_grappler_optimizer(input_gdef, output_nodes):
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(input_gdef, name="")
    grappler_meta_graph_def = tf.train.export_meta_graph(
        graph_def=graph.as_graph_def(add_shapes=True), graph=graph)

    _to_bytes = lambda s: s.encode("utf-8", errors="surrogateescape")
    output_collection = meta_graph_pb2.CollectionDef()
    output_list = output_collection.node_list.value
    for i in output_nodes:
        if isinstance(i, tf.Tensor):
            output_list.append(_to_bytes(i.name))
        else:
            output_list.append(_to_bytes(i))
    # TODO(laigd): use another key as the outputs are really not train_op.
    grappler_meta_graph_def.collection_def["train_op"].CopyFrom(
        output_collection)

    session_config = tf.ConfigProto()
    session_config = ngraph_bridge.update_config(session_config)
    output_gdef = tf_optimizer.OptimizeGraph(
        session_config, grappler_meta_graph_def, graph_id=b"tf_graph")
    return output_gdef


def get_gdef_from_savedmodel(export_dir):
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                   export_dir)
        return sess.graph.as_graph_def()


def get_gdef_from_protobuf(pb_filename):
    graph_def = tf.GraphDef()
    if pb_filename.endswith("pbtxt"):
        with open(pb_filename, "r") as f:
            text_format.Merge(f.read(), graph_def)
    else:
        with open(pb_filename, "rb") as f:
            graph_def.ParseFromString(f.read())
    return graph_def


def check_graph_validity(gdef):
    # Assuming that the input graph has not already been processed by ngraph
    # TODO: add other checks for other types on NG ops
    not_already_processed = all(
        [i.op is not 'NGraphEncapsulate' for i in gdef.node])
    # Assume it is an inference ready graph
    no_variables = all(['Variable' not in i.op for i in gdef.node])
    return not_already_processed and no_variables


def get_gdef(format, location):
    gdef = {
        'savedmodel': get_gdef_from_savedmodel,
        'pbtxt': get_gdef_from_protobuf,
        'pb': get_gdef_from_protobuf
    }[format](location)
    assert check_graph_validity(gdef)
    return gdef


def prepare_argparser(formats):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='''
    Tool to convert TF graph into a ngraph enabled graph
    Sample usage:
    Command line:
    python tf2ngraph.py --inputsavedmodel test_graph_SM --outnodes out_node --outputpbtxt test_graph_SM_mod.pbtxt
    python tf2ngraph.py --inputpbtxt test_graph_SM.pbtxt --outnodes out_node --outputpbtxt test_graph_SM_mod.pbtxt
    or:
    functional api
    from tf2ngraph import convert
    convert('savedmodel', 'test_graph_SM' , 'pbtxt', 'test_graph_SM_mod.pbtxt', ['out_node'])
    convert('pbtxt', 'test_graph_SM.pbtxt' , 'pbtxt', 'test_graph_SM_mod.pbtxt', ['out_node'])
    ''')
    in_out_groups = [parser.add_argument_group(i) for i in ['input', 'output']]
    for grp in in_out_groups:
        inp_out_group = grp.add_mutually_exclusive_group()
        for format in formats[grp.title]:
            opt_name = grp.title + format
            inp_out_group.add_argument(
                "--" + opt_name, help="Location of " + grp.title + " " + format)
    # Note: no other option must begin with "input" or "output"
    parser.add_argument(
        "--outnodes", help="Comma separated list of output nodes")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    return parser.parse_args()


def filter_dict(prefix, dictionary):
    assert prefix in ['input', 'output']
    current_format = list(
        filter(lambda x: x.startswith(prefix) and dictionary[x] is not None,
               dictionary))
    assert len(current_format) == 1, "Got " + str(
        len(current_format)) + " input formats, expected only 1"
    # [len(prefix):] deletes the initial "input" in the string
    stripped = current_format[0][len(prefix):]
    assert stripped in allowed_formats[
        prefix], "Got " + prefix + " format = " + stripped + " but only support " + str(
            allowed_formats[prefix])
    return (stripped, dictionary[prefix + stripped])


def save_gdef_to_savedmodel(gdef, location):
    builder = tf.saved_model.builder.SavedModelBuilder(location)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(gdef, name="")
        with tf.Session(graph=graph) as sess:
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.TRAINING])
            builder.add_meta_graph([tf.saved_model.tag_constants.SERVING],
                                   strip_default_attrs=True)
        builder.save()


def save_gdef_to_protobuf(gdef, location, as_text):
    tf.io.write_graph(
        gdef,
        os.path.dirname(location),
        os.path.basename(location),
        as_text=as_text)


def save_model(gdef, format, location):
    return {
        'savedmodel': save_gdef_to_savedmodel,
        'pbtxt': partial(save_gdef_to_protobuf, as_text=True),
        'pb': partial(save_gdef_to_protobuf, as_text=False)
    }[format](gdef, location)


def attach_device(gdef):
    for n in gdef.node:
        n.device = "/device:CPU:0"


allowed_formats = {
    "input": ['savedmodel', 'pbtxt', 'pb'],
    "output": ['savedmodel', 'pbtxt', 'pb']
}


def convert(inp_format, inp_loc, out_format, out_loc, outnodes):
    """Functional api for converting TF models by inserting ngraph nodes.
    Sample usage:
    from tf2ngraph import convert
    convert('savedmodel', 'test_graph_SM' , 'pbtxt', 'test_graph_SM_mod.pbtxt', ['out_node'])
    convert('pbtxt', 'test_graph_SM.pbtxt' , 'pbtxt', 'test_graph_SM_mod.pbtxt', ['out_node'])

    Parameters:
    inp_format (string): 'savedmodel', 'pbtxt', 'pb'
    inp_loc (string): Location of input file or folder (in case of savedmodel)
    out_format (string): 'savedmodel', 'pbtxt', 'pb'
    out_loc (string): Location of output file or folder (in case of savedmodel)
    outnodes (iterable of strings): names of output nodes

    Returns: void
   """
    assert inp_format in allowed_formats['input']
    assert out_format in allowed_formats['output']
    assert ngraph_bridge.is_grappler_enabled()
    input_gdef = get_gdef(inp_format, inp_loc)
    attach_device(input_gdef)
    output_gdef = run_ngraph_grappler_optimizer(input_gdef, outnodes)
    save_model(output_gdef, out_format, out_loc)


def main():
    """ Entry point of command line api for converting TF models by inserting ngraph nodes.
    Sample usage:
    python tf2ngraph.py --inputsavedmodel test_graph_SM --outnodes out_node --outputpbtxt test_graph_SM_mod.pbtxt
    python tf2ngraph.py --inputpbtxt test_graph_SM.pbtxt --outnodes out_node --outputpbtxt test_graph_SM_mod.pbtxt
    """
    args = prepare_argparser(allowed_formats)
    inp_format, inp_loc = filter_dict("input", args.__dict__)
    out_format, out_loc = filter_dict("output", args.__dict__)
    outnodes = args.outnodes.split(',')
    convert(inp_format, inp_loc, out_format, out_loc, outnodes)
    print('Converted the model. Exiting now')


if __name__ == '__main__':
    main()
