# ==============================================================================
#  Copyright 2019-2020 Intel Corporation
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
tf.compat.v1.disable_eager_execution()
from google.protobuf import text_format
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.python.grappler import tf_optimizer
import ngraph_bridge
import os
import sys
import json
from functools import partial
from tensorflow.python.framework.function_def_to_graph import function_def_to_graph


class Tf2ngraphJson(object):

    @staticmethod
    def allowed_fields():
        return set(["shape_hints", "backend_optional_params"])

    @staticmethod
    def assert_type(obj, expected_type, tag):
        exit_on_error(
            type(obj) == expected_type, "Expected " + tag + " to be " +
            str(expected_type) + " but got " + str(type(obj)))

    @staticmethod
    def check_shape_hints(shape_hints):
        Tf2ngraphJson.assert_type(shape_hints, type([]), 'shape_hints')
        for item in shape_hints:
            Tf2ngraphJson.assert_type(item, type({}),
                                      'each element of the shape_hints list')
            for k in item:
                Tf2ngraphJson.assert_type(
                    k, type(""), 'the keys of dictionaries in shape_hints list')
                Tf2ngraphJson.assert_type(
                    item[k], type([]),
                    'the values of dictionaries in shape_hints list')

    @staticmethod
    def check_optional_params(opt_params):
        for optional_attr in opt_params:
            Tf2ngraphJson.assert_type(opt_params[optional_attr], type(""),
                                      'keys of backend_optional_params')
            Tf2ngraphJson.assert_type(optional_attr, type(""),
                                      'values of backend_optional_params')

    @staticmethod
    def parse_json(json_name):
        if json_name == '':
            return {}, []
        optional_backend_params = {}
        shape_hints = []
        with open(json_name) as f:
            dct = json.load(f)
            for k in dct:
                if k == 'shape_hints':
                    Tf2ngraphJson.check_shape_hints(dct[k])
                    shape_hints = dct[k]
                elif k == 'backend_optional_params':
                    Tf2ngraphJson.check_optional_params(dct[k])
                    optional_backend_params = dct[k]
                else:
                    exit_on_error(
                        False,
                        "Expected keys of config json file to be: " + \
                        str(allowed_fields())) + ", but got " + str(k)
        return optional_backend_params, shape_hints

    @staticmethod
    def dump_json(json_name, optional_params=None, shape_hints=None):
        optional_params = {} if optional_params is None else optional_params
        shape_hints = [] if shape_hints is None else shape_hints
        Tf2ngraphJson.check_optional_params(optional_params)
        Tf2ngraphJson.check_shape_hints(shape_hints)
        dict_to_dump = {"backend_optional_params": optional_params}
        if len(shape_hints) > 0:
            dict_to_dump["shape_hints"] = shape_hints
        with open(json_name, 'w') as fp:
            json.dump(dict_to_dump, fp)


# This function controls how errors are handled.
# For developers/debugging set assert_on_failure to True
def exit_on_error(success, error_message, assert_on_failure=False):
    if not success:
        if assert_on_failure:
            assert success, error_message
        else:
            sys.stderr.write("\n" + error_message + "\n")
            sys.exit(1)


def update_config_to_include_custom_config(config, backend, device_id,
                                           backend_optional_params, shape_hints,
                                           do_aot):
    rewriter_options = rewriter_config_pb2.RewriterConfig()
    rewriter_options.meta_optimizer_iterations = (
        rewriter_config_pb2.RewriterConfig.ONE)
    rewriter_options.min_graph_nodes = -1
    rewriter_options.fail_on_optimizer_errors = True
    ngraph_optimizer = rewriter_options.custom_optimizers.add()
    ngraph_optimizer.name = "ngraph-optimizer"
    ngraph_optimizer.parameter_map["ngraph_backend"].s = backend.encode()
    ngraph_optimizer.parameter_map["device_id"].s = device_id.encode()
    for k in backend_optional_params:
        ngraph_optimizer.parameter_map[k].s = backend_optional_params[k].encode(
        )
    # Attach shape hints
    for hint_id, shape_hint in enumerate(shape_hints):
        shape_hint_name = "shape_hint_" + str(hint_id)
        ngraph_optimizer.parameter_map[
            shape_hint_name].func.name = shape_hint_name.encode()
        ngraph_optimizer.parameter_map[shape_hint_name].func.attr.get_or_create(
            'hint_body').func.name = b'hint_body'
        for node_name in shape_hint:  # TODO: verify that node names passed in shape hints are valid node names present in the graph
            ngraph_optimizer.parameter_map[
                shape_hint_name].func.attr.get_or_create(
                    'hint_body').func.attr.get_or_create(
                        node_name).tensor.int_val.extend(shape_hint[node_name])
    # Attach aot request
    ngraph_optimizer.parameter_map["aot_requested"].s = str(
        ("0", "1")[do_aot]).encode()
    config.MergeFrom(
        tf.compat.v1.ConfigProto(
            graph_options=tf.compat.v1.GraphOptions(
                rewrite_options=rewriter_options)))
    return config


def run_ngraph_grappler_optimizer(input_gdef, output_nodes, ng_backend,
                                  device_id, backend_optional_params,
                                  shape_hints, do_aot):
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(input_gdef, name="")
    grappler_meta_graph_def = tf.compat.v1.train.export_meta_graph(
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

    session_config = tf.compat.v1.ConfigProto()
    # Pass backend and backend_optional_params to grappler through rewriter config by updating the config
    # TODO: move update_config_to_include_custom_config to ngraph_bridge
    session_config = update_config_to_include_custom_config(
        session_config, ng_backend, device_id, backend_optional_params,
        shape_hints, do_aot)
    try:
        output_gdef = tf_optimizer.OptimizeGraph(
            session_config, grappler_meta_graph_def, graph_id=b"tf_graph")
    except Exception as e:
        exit_on_error(False, e.message)
    return output_gdef


def get_gdef_from_savedmodel(export_dir):
    with tf.compat.v1.Session(graph=tf.compat.v1.Graph()) as sess:
        tf.compat.v1.saved_model.loader.load(
            sess, [tf.compat.v1.saved_model.tag_constants.SERVING], export_dir)
        return sess.graph.as_graph_def()


def get_gdef_from_protobuf(pb_filename):
    graph_def = tf.compat.v1.GraphDef()
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
    already_processed = any([i.op is 'NGraphEncapsulate' for i in gdef.node])
    # Assume it is an inference ready graph
    has_variables = any(['Variable' in i.op for i in gdef.node])
    return already_processed, has_variables


def get_gdef(format, location):
    gdef = {
        'savedmodel': get_gdef_from_savedmodel,
        'pbtxt': get_gdef_from_protobuf,
        'pb': get_gdef_from_protobuf
    }[format](location)
    already_processed, has_variables = check_graph_validity(gdef)
    if already_processed or has_variables:
        err_string = ["Graph at " + location + " is not acceptable because:"]
        if already_processed:
            err_string.append(
                "It already contains encapsulate ops (and migth not need running through tf2ngraph again)"
            )
        if has_variables:
            err_string.append(
                "It contains Variables (please freeze the graph to convert variables to constant)"
            )
        exit_on_error(False, '\n'.join(err_string))
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
    python tf2ngraph.py --input_pbtxt ../test/test_axpy.pbtxt --output_nodes add --output_pbtxt axpy_ngraph.pbtxt --ng_backend INTERPRETER --config_file sample_optional_params_and_shape_hints.json --precompile
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
        required=False,
        default=None)
    parser.add_argument(
        "--ng_backend", default='CPU', help="Ngraph backend. Eg, NNPI")
    parser.add_argument("--device_id", default='', help="Device id. Eg, 0")
    parser.add_argument(
        "--config_file",
        default='',
        help=
        "Json file that contains optional backend configuration settings and shape hints"
    )
    parser.add_argument(
        "--precompile",
        action='store_true',
        help=
        "Perform precompilation to embed the ngraph executable in the dumped TF graph"
    )
    parser.add_argument(
        "--save_ng_clusters",
        action='store_true',
        help="Saves the TF subgraphs that each ngraph encapsulate replaces")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    return parser.parse_args()


def filter_dict(prefix, dictionary):
    exit_on_error(prefix in ['input', 'output'],
                  "Expected prefix to be 'input' or 'output' but got " + prefix)
    current_format = list(
        filter(
            lambda x: x.startswith(prefix + '_') and dictionary[x] is not None
            and 'nodes' not in x, dictionary))
    exit_on_error(len(current_format) == 1, "Got " + str(len(current_format)) + \
    " " + prefix + " formats, expected exactly 1 " + prefix + \
    " format. Please add one of --" + prefix + "_pb or --" + prefix + \
    "_pbtxt or --" + prefix + "_savedmodel and pass the " + prefix + " model location")
    # [len(prefix):] deletes the initial "input" in the string
    stripped = current_format[0][len(prefix):].lstrip('_')
    exit_on_error(
        stripped in allowed_formats[prefix], "Got " + prefix + " format = " +
        stripped + " but only support " + str(allowed_formats[prefix]))
    return (stripped, dictionary[prefix + '_' + stripped])


def save_gdef_to_savedmodel(gdef, location):
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(location)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(gdef, name="")
        with tf.compat.v1.Session(graph=graph) as sess:
            builder.add_meta_graph_and_variables(
                sess, [tf.compat.v1.saved_model.tag_constants.TRAINING])
            builder.add_meta_graph(
                [tf.compat.v1.saved_model.tag_constants.SERVING],
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
            device_id, backend_optional_params, shape_hints, do_aot,
            save_ng_clusters):
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
    exit_on_error(
        inp_format in allowed_formats['input'], 'Unsupported input format ' +
        inp_format + ". Supported formats: " + str(allowed_formats['input']))
    exit_on_error(
        out_format in allowed_formats['output'], 'Unsupported output format ' +
        out_format + ". Supported formats: " + str(allowed_formats['output']))
    exit_on_error(
        ngraph_bridge.is_grappler_enabled(),
        "ngraph-bridge is not built with grappler enabled, hence tf2ngraph is not supported."
    )
    input_gdef = get_gdef(inp_format, inp_loc)
    attach_device(input_gdef)
    output_gdef = run_ngraph_grappler_optimizer(
        input_gdef, output_nodes, ng_backend, device_id,
        backend_optional_params, shape_hints, do_aot)
    if save_ng_clusters:
        for fn in output_gdef.library.function:
            tf.io.write_graph(
                function_def_to_graph(fn).as_graph_def(),
                '.',
                fn.signature.name + '.pbtxt',
                as_text=True)
    save_model(output_gdef, out_format, out_loc)


def sanitize_node_name(node_name):
    '''
    Given an input to a node in the graph def clean it to find the node name
    '''
    # get rid of caret indicating control edge (^name -> name)
    if node_name.startswith('^'):
        node_name = node_name[1:]

    # get rid of output slot (name:0 -> name)
    split_colon = node_name.split(':')
    if len(split_colon) == 1 or len(split_colon) == 2:
        return split_colon[0]
    else:
        exit_on_error(False, "Expected node name to have <= 1 colons. " + \
        "TODO: Handle case with > 1 colons")


def get_possible_output_node_names(graph_def):
    '''
    Nodes which do not appear in the inputs of other nodes
    are returned as possible output nodes.
    '''
    nodes_which_appear_at_inputs = set()
    for n in graph_def.node:
        # the list comprehension converts a
        # google.protobuf.pyext._message.RepeatedScalarContainer to a list of strings
        nodes_which_appear_at_inputs.update(
            [sanitize_node_name(i) for i in n.input])

    all_node_names = {n.name for n in graph_def.node}
    possible_outputs = all_node_names.difference(nodes_which_appear_at_inputs)
    name_type_map = get_name_type_map(graph_def)
    return {k: name_type_map[k] for k in possible_outputs}


def get_name_type_map(graph_def):
    return {n.name: n.op for n in graph_def.node}


def infer_output_nodes(inp_format, inp_loc):
    '''
    Try to read output names from savedmodel's signature_def
    or try to guess the outputs from the graphdef
    '''
    if inp_format == 'savedmodel':
        with tf.compat.v1.Session(graph=tf.compat.v1.Graph()) as sess:
            # load the saved model
            imported = tf.compat.v1.saved_model.load(
                sess,
                tags=[tf.compat.v1.saved_model.tag_constants.SERVING],
                export_dir=inp_loc)
            try:
                # Check if the saved model has outputs specified
                output_info = imported.signature_def[
                    'serving_default'].outputs.values()
                saved_model_has_out_name = True
            except:
                saved_model_has_out_name = False
            if saved_model_has_out_name:
                # the list comprehension gets the names
                # from the output_info of type collections.abc.ValuesView
                possible_outputs = [
                    sanitize_node_name(i.name) for i in output_info
                ]
                node_name_type_dict = get_name_type_map(imported.graph_def)
                return {k: node_name_type_dict[k] for k in possible_outputs}
            else:
                # Outputs not specified in the saved model
                # Hence using get_possible_output_node_names
                return get_possible_output_node_names(sess.graph_def)
    elif inp_format == 'pbtxt' or inp_format == 'pb':
        return get_possible_output_node_names(get_gdef_from_protobuf(inp_loc))
    else:
        exit_on_error(
            False,
            "inp_format expected to be pb, pbtxt or savedmodel, but found ",
            inp_format)


def main():
    """ Entry point of command line api for converting TF models by inserting ngraph nodes.
    Sample usage:
    python tf2ngraph.py --inputsavedmodel test_graph --output_nodes out_node --outputpbtxt test_graph_ngraph.pbtxt --ng_backend NNPI:0
    python tf2ngraph.py --inputpbtxt test_graph.pbtxt --output_nodes out_node --outputpbtxt test_graph_ngraph.pbtxt --ng_backend NNPI:0
    """
    args = prepare_argparser(allowed_formats)
    inp_format, inp_loc = filter_dict("input", args.__dict__)
    out_format, out_loc = filter_dict("output", args.__dict__)
    if args.output_nodes is None:
        possible_out_nodes = infer_output_nodes(inp_format, inp_loc)
        print(
            "\nAnalyzed graph for possible list of output nodes. " + \
            "Please supply one or more output node in --output_nodes"
        )
        for out_node in possible_out_nodes:
            print("Name: `" + out_node + "` Type: `" +
                  possible_out_nodes[out_node] + "`")
        print()
        exit_on_error(False, "No output node name provided in --output_nodes")
    else:
        output_nodes = args.output_nodes.split(',')
    backend_optional_params, shape_hints = Tf2ngraphJson.parse_json(
        args.config_file)
    convert(inp_format, inp_loc, out_format, out_loc, output_nodes,
            args.ng_backend, args.device_id, backend_optional_params,
            shape_hints, args.precompile, args.save_ng_clusters)
    print('Converted the model. Exiting now')


if __name__ == '__main__':
    main()

    # TODO what happens if same shape is passed twice
