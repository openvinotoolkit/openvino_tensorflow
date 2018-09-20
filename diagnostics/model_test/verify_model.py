import tensorflow as tf
import argparse
import numpy as np
import ngraph
import json
import os


def calculate_output(param_dict, select_device, input_example):
    """Calculate the output of the imported frozen graph given the input.

    Load the graph def from frozen_graph_file on selected device, then get the tensors based on the input and output name from the graph,
    then feed the input_example to the graph and retrieves the output vector.

    Args:
    param_dict: The dictionary contains all the user-input data in the json file.
    select_device: "NGRAPH" or "CPU".
    input_example: A map with key is the name of the input tensor, and value is the random generated example

    Returns:
        The output vector obtained from running the input_example through the graph.
    """
    frozen_graph_filename = param_dict["frozen_graph_location"]
    output_tensor_name = param_dict["output_tensor_name"]

    if not tf.gfile.Exists(frozen_graph_filename):
        raise Exception("Input graph file '" + frozen_graph_filename +
                        "' does not exist!")

    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    if select_device == 'CPU':
        ngraph.disable()
    else:
        # run on NGRAPH
        if not ngraph.is_enabled():
            ngraph.enable()

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # Create the tensor to its corresponding example map
    tensor_to_example_map = {}
    for item in input_example:
        t = graph.get_tensor_by_name(item)
        tensor_to_example_map[t] = input_example[item]

    #input_placeholder = graph.get_tensor_by_name(input_tensor_name)
    output_tensor = graph.get_tensor_by_name(output_tensor_name)

    config = tf.ConfigProto(
        allow_soft_placement=True,
        # log_device_placement=True,
        inter_op_parallelism_threads=1)

    with tf.Session(graph=graph, config=config) as sess:
        output_tensor = sess.run(output_tensor, feed_dict=tensor_to_example_map)
        return output_tensor


def calculate_norm(ngraph_output, tf_output, desired_norm):
    """Calculate desired_norm between vectors.

    Calculate the L1/L2/inf norm between the NGRAPH and tensorflow output vectors.

    Args:
        ngraph_output: The output vector generated from NGRAPH graph.
        tf_output: The output vector generated from tensorflow graph.
        desired_norm: L1/L2/inf norm. 

    Returns:
        Calculated norm between the vectors.

    Raises:
        Exception: If the dimension of the two vectors mismatch.
    """
    if (ngraph_output.shape != tf_output.shape):
        raise Exception('ngraph output and tf output dimension mismatch')

    ngraph_output_squeezed = np.squeeze(ngraph_output)
    tf_output_squeezed = np.squeeze(tf_output)

    ngraph_output_flatten = ngraph_output_squeezed.flatten()
    tf_output_flatten = tf_output_squeezed.flatten()

    factor = np.prod(ngraph_output_squeezed.shape)

    if desired_norm not in [1, 2, np.inf]:
        raise Exception('Only L2, L2, and inf norms are supported')

    return np.linalg.norm((ngraph_output_flatten - tf_output_flatten),
                          desired_norm)


def parse_json():
    """
        Parse the user input json file.

        Returns:
            A dictionary contains all the parsed parameters.
    """
    with open(os.path.abspath(args.json_file)) as f:
        parsed_json = json.load(f)
        return parsed_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_file", type=str, help="Model details in json format")

    args = parser.parse_args()

    if args.json_file is None:
        raise ValueError("Supply a json file to start")

    parameters = parse_json()

    # Generate random input based on input_dimension
    np.random.seed(100)
    input_dimension = parameters["input_dimension"]
    input_tensor_name = parameters["input_tensor_name"]
    bs = int(parameters["batch_size"])

    assert len(input_dimension) == len(
        input_tensor_name
    ), "input_tensor_name dimension should match input_dimension in json file"

    # Matches the input tensors name with its required dimensions
    input_tensor_dim_map = {}
    for (dim, name) in zip(input_dimension, input_tensor_name):
        random_input = np.random.random_sample([bs] + dim)
        input_tensor_dim_map[name] = random_input

    # Run the model on tensorflow
    result_tf_graph = calculate_output(parameters, "CPU", input_tensor_dim_map)
    # Run the model on ngraph
    result_ngraph = calculate_output(parameters, "NGRAPH", input_tensor_dim_map)

    l1_norm = calculate_norm(result_ngraph, result_tf_graph, 1)
    l2_norm = calculate_norm(result_ngraph, result_tf_graph, 2)
    inf_norm = calculate_norm(result_ngraph, result_tf_graph, np.inf)

    l1_norm_threshold = parameters["l1_norm_threshold"]
    l2_norm_threshold = parameters["l2_norm_threshold"]
    inf_norm_threshold = parameters["inf_norm_threshold"]

    if l1_norm > l1_norm_threshold:
        print("The L1 norm %f is greater than the threshold %f " %
              (l1_norm, l1_norm_threshold))
    else:
        print("L1 norm test passed")

    if l2_norm > l2_norm_threshold:
        print("The L2 norm %f is greater than the threshold %f " %
              (l2_norm, l2_norm_threshold))
    else:
        print("L2 norm test passed")

    if inf_norm > inf_norm_threshold:
        print("The inf norm %f is greater than the threshold %f " %
              (inf_norm, inf_norm_threshold))
    else:
        print("inf norm test passed")
