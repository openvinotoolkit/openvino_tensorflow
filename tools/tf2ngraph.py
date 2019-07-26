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
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.grappler import tf_optimizer
import ngraph_bridge
import os
import sys
from functools import partial


def parse_extra_params_string(raw_extra_params):
    raw_extra_params = raw_extra_params.strip(' ')
    assert raw_extra_params[0] == '{' and raw_extra_params[
        -1] == '}', "Expected extra_params string to be a dictionary beginning with { and ending with }"
    raw_extra_params_contents = raw_extra_params[1:-1].strip(' ')
    extra_params_dict = {}
    if len(raw_extra_params_contents) == 0:
        return extra_params_dict
    # could have used eval(extra_params_string), but then the string would have to be the cumbersome {\"abc\":1}
    # and not {"abc":1} or {abc:1}. Hence explicity parsing the string without using eval
    for key_val in raw_extra_params_contents.split(','):
        key_val = key_val.strip(' ')
        try:
            key, val = key_val.split(':')
            extra_params_dict[key.strip(' ')] = val.strip(' ')
        except Exception as e:
            raise type(
                e
            )(e.message +
              'Got an entry in extra_params, that is an invalid entry for a python dictionary: '
              + key_val)
    return extra_params_dict


def update_config_to_include_custom_config(config, backend, device_id,
                                           extra_params):
    rewriter_options = rewriter_config_pb2.RewriterConfig()
    rewriter_options.meta_optimizer_iterations = (
        rewriter_config_pb2.RewriterConfig.ONE)
    rewriter_options.min_graph_nodes = -1
    ngraph_optimizer = rewriter_options.custom_optimizers.add()
    ngraph_optimizer.name = "ngraph-optimizer"
    ngraph_optimizer.parameter_map["ngraph_backend"].s = backend.encode()
    ngraph_optimizer.parameter_map["device_id"].s = device_id.encode()
    for k in extra_params:
        ngraph_optimizer.parameter_map[k].s = extra_params[k].encode()
    config.MergeFrom(
        tf.ConfigProto(
            graph_options=tf.GraphOptions(rewrite_options=rewriter_options)))
    return config


def run_ngraph_grappler_optimizer(input_gdef, output_nodes, ng_backend,
                                  device_id, extra_params):
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
    # Pass backend and extra backend params to grappler through rewriter config by updating the config
    # TODO: move update_config_to_include_custom_config to ngraph_bridge
    session_config = update_config_to_include_custom_config(
        session_config, ng_backend, device_id, extra_params)
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
    python tf2ngraph.py --input_savedmodel resnet_model_location --output_nodes out_node --output_pbtxt resnet_ngraph.pbtxt
    python tf2ngraph.py --input_pbtxt mobilenet.pbtxt --output_nodes out_node --output_pbtxt mobilenet_ngraph.pbtxt
    python tf2ngraph.py --input_pb inception_v3_2016_08_28_frozen.pb --output_nodes InceptionV3/Predictions/Reshape_1 --output_pb inception_v3_2016_08_28_frozen_ngraph.pb
    python tf2ngraph.py --input_pbtxt ../test/test_axpy.pbtxt --output_nodes add --output_pbtxt axpy_ngraph.pbtxt --ng_backend CPU
    ''')
    in_out_groups = [
        parser.add_argument_group(i, j) for i, j in zip(
            ['input', 'output'], ['Input formats', 'Output formats'])
    ]
    for grp in in_out_groups:
        inp_out_group = grp.add_mutually_exclusive_group()
        for format in formats[grp.title]:
            opt_name = grp.title + '_' + format
            inp_out_group.add_argument(
                "--" + opt_name, help="Location of " + grp.title + " " + format)
    # Note: no other option must begin with "input" or "output"
    parser.add_argument(
        "--output_nodes",
        help=
        "Comma separated list of output nodes. Output nodes can be found " \
        "by manual inspection of the graph, prior knowledge or running the " \
        "summarize_graph tool provided by Tensorflow",
        required=True)
    parser.add_argument(
        "--ng_backend", default='CPU', help="Ngraph backend. Eg, NNPI")
    parser.add_argument("--device_id", default='', help="Device id. Eg, 0")
    parser.add_argument(
        "--extra_params",
        default='{}',
        help=
        "Other params that the backend needs in the form of a dictionary. Eg, {max_cores: 4}."
    )
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    return parser.parse_args()


def filter_dict(prefix, dictionary):
    assert prefix in ['input', 'output']
    current_format = list(
        filter(
            lambda x: x.startswith(prefix + '_') and dictionary[x] is not None
            and 'nodes' not in x, dictionary))
    assert len(current_format) == 1, "Got " + str(
        len(current_format)) + " " + prefix + " formats, expected only 1"
    # [len(prefix):] deletes the initial "input" in the string
    stripped = current_format[0][len(prefix):].lstrip('_')
    assert stripped in allowed_formats[
        prefix], "Got " + prefix + " format = " + stripped + " but only support " + str(
            allowed_formats[prefix])
    return (stripped, dictionary[prefix + '_' + stripped])


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


def convert(inp_format, inp_loc, out_format, out_loc, output_nodes, ng_backend,
            device_id, extra_params):
    """Functional api for converting TF models by inserting ngraph nodes.
    Sample usage:
    from tf2ngraph import convert
    convert('savedmodel', 'test_graph' , 'pbtxt', 'test_graph_ngraph.pbtxt', ['out_node'])
    convert('pbtxt', 'test_graph.pbtxt' , 'pbtxt', 'test_graph_ngraph.pbtxt', ['out_node'])

    Parameters:
    inp_format (string): 'savedmodel', 'pbtxt', 'pb'
    inp_loc (string): Location of input file or folder (in case of savedmodel)
    out_format (string): 'savedmodel', 'pbtxt', 'pb'
    out_loc (string): Location of output file or folder (in case of savedmodel)
    output_nodes (iterable of strings): names of output nodes

    Returns: void
   """
    assert inp_format in allowed_formats['input']
    assert out_format in allowed_formats['output']
    assert ngraph_bridge.is_grappler_enabled()
    input_gdef = get_gdef(inp_format, inp_loc)
    attach_device(input_gdef)
    output_gdef = run_ngraph_grappler_optimizer(
        input_gdef, output_nodes, ng_backend, device_id, extra_params)
    save_model(output_gdef, out_format, out_loc)


def main():
    """ Entry point of command line api for converting TF models by inserting ngraph nodes.
    Sample usage:
    python tf2ngraph.py --inputsavedmodel test_graph --output_nodes out_node --outputpbtxt test_graph_ngraph.pbtxt --ng_backend NNPI:0
    python tf2ngraph.py --inputpbtxt test_graph.pbtxt --output_nodes out_node --outputpbtxt test_graph_ngraph.pbtxt --ng_backend NNPI:0
    """
    args = prepare_argparser(allowed_formats)
    inp_format, inp_loc = filter_dict("input", args.__dict__)
    out_format, out_loc = filter_dict("output", args.__dict__)
    output_nodes = args.output_nodes.split(',')
    extra_params = parse_extra_params_string(args.extra_params)
    convert(inp_format, inp_loc, out_format, out_loc, output_nodes,
            args.ng_backend, args.device_id, extra_params)
    print('Converted the model. Exiting now')


if __name__ == '__main__':
    main()
