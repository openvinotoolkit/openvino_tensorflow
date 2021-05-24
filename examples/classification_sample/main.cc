/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
//-----------------------------------------------------------------------------
// NOTE: This file is taken from tensorflow/examples/label_image/main.cc
// and modified for this example:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/main.cc
//-----------------------------------------------------------------------------
/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
// A minimal but useful C++ example showing how to load an Imagenet-style object
// recognition TensorFlow model, prepare input images for it, run them through
// the graph, and interpret the results.
//
// It's designed to have as few dependencies and be as clear as possible, so
// it's more verbose than it could be in production code. In particular, using
// auto for the types of a lot of the returned values from TensorFlow calls can
// remove a lot of boilerplate, but I find the explicit types useful in sample
// code to make it simple to look up the classes involved.
//
// To use it, compile and then run in a working directory with the
// learning/brain/tutorials/label_image/data/ folder below it, and you should
// see the top five labels for the example Lena image output. You can then
// customize it to use your own models or images by changing the file names at
// the top of the main() function.
//
// The googlenet_graph.pb file included by default is created from Inception.
//
// Note that, for GIF inputs, to reuse existing code, only single-frame ones
// are supported.

#include <thread>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/public/version.h"
#if (TF_MAJOR_VERSION >= 2) && (TF_MINOR_VERSION > 2)
#include "tensorflow/core/common_runtime/graph_constructor.h"
#else
#include "tensorflow/core/graph/graph_constructor.h"
#endif
#include "openvino_tensorflow/api.h"
#include "openvino_tensorflow/ovtf_timer.h"
#include "openvino_tensorflow/ovtf_utils.h"
#include "openvino_tensorflow/version.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

using namespace std;
using tensorflow::Flag;
using tensorflow::Status;
using tensorflow::Tensor;

extern tensorflow::Status LoadGraph(
    const string& graph_file_name,
    std::unique_ptr<tensorflow::Session>* session);

extern tensorflow::Status ReadTensorFromImageFile(
    const string& file_name, const int input_height, const int input_width,
    const float input_mean, const float input_std,
    std::vector<tensorflow::Tensor>* out_tensors);
extern tensorflow::Status PrintTopLabels(
    const std::vector<tensorflow::Tensor>& outputs,
    const string& labels_file_name);
extern tensorflow::Status CheckTopLabel(
    const std::vector<tensorflow::Tensor>& outputs, int expected,
    bool* is_expected);

// Prints the available backends
void PrintAvailableBackends() {
  // Get the list of backends
  auto supported_backends =
      tensorflow::openvino_tensorflow::api::ListBackends();
  vector<string> backends(supported_backends.begin(), supported_backends.end());

  cout << "Available backends: " << endl;
  for (auto& backend_name : backends) {
    cout << "Backend: " << backend_name << std::endl;
  }
}

void PrintVersion() {
  // Tensorflow version info
  std::cout << "Tensorflow version: "
            << tensorflow::openvino_tensorflow::tf_version() << std::endl;
  // Openvino integration with TensorFlow info
  std::cout << "OpenVINO integration with TensorFlow version: "
            << tensorflow::openvino_tensorflow::version() << std::endl;
  std::cout << "CXX11_ABI Used: "
            << tensorflow::openvino_tensorflow::cxx11_abi_flag() << std::endl;
  std::cout << "Grappler Enabled? "
            << (tensorflow::openvino_tensorflow::is_grappler_enabled()
                    ? std::string("Yes")
                    : std::string("No"))
            << std::endl;
  PrintAvailableBackends();
}

int main(int argc, char** argv) {
  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than inception_v3, then you'll need to update these.
  string image_file = "examples/data/grace_hopper.jpg";
  string graph = "examples/data/inception_v3_2016_08_28_frozen.pb";
  string labels_file = "examples/data/imagenet_slim_labels.txt";
  int input_width = 299;
  int input_height = 299;
  float input_mean = 0;
  float input_std = 255;
  string input_layer = "input";
  string output_layer = "InceptionV3/Predictions/Reshape_1";
  bool self_test = false;
  string root_dir = "";
  string backend_name = "CPU";

  std::vector<tensorflow::Flag> flag_list = {
      Flag("image", &image_file, "image to be processed"),
      Flag("graph", &graph, "graph to be executed"),
      Flag("labels", &labels_file, "name of file containing labels"),
      // Flag("label_index", &label_index, "Index of the expected label"),
      Flag("input_width", &input_width, "resize image to this width in pixels"),
      Flag("input_height", &input_height,
           "resize image to this height in pixels"),
      Flag("input_mean", &input_mean, "scale pixel values to this mean"),
      Flag("input_std", &input_std, "scale pixel values to this std deviation"),
      Flag("input_layer", &input_layer, "name of input layer"),
      Flag("output_layer", &output_layer, "name of output layer"),
      Flag("self_test", &self_test, "run a self test"),
      Flag("root_dir", &root_dir,
           "interpret image and graph file names relative to this directory"),
      Flag("backend", &backend_name, "backend option. Default is CPU")};

  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    std::cout << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    std::cout << "Error: Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  std::cout << "Component versions used\n";
  PrintVersion();

  // Enable differnt backends(CPU/GPU/MYRIAD/HDDL) to run the network.
  tensorflow::openvino_tensorflow::api::SetBackend(backend_name);

  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  std::vector<Tensor> resized_tensors;
  string image_path = tensorflow::io::JoinPath(root_dir, image_file);
  Status read_tensor_status =
      ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
                              input_std, &resized_tensors);
  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    return -1;
  }
  const Tensor& resized_tensor = resized_tensors[0];

  //  Warm up
  std::vector<Tensor> outputs;
  tensorflow::openvino_tensorflow::Timer compilation_timer;
  Status run_status = session->Run({{input_layer, resized_tensor}},
                                   {output_layer}, {}, &outputs);
  compilation_timer.Stop();
  cout << "Compilation Time in ms: " << compilation_timer.ElapsedInMS() << endl;

  if (!run_status.ok()) {
    LOG(ERROR) << "Compiling model failed: " << run_status;
    return -1;
  }

  //  Run
  tensorflow::openvino_tensorflow::Timer inference_timer;
  run_status = session->Run({{input_layer, resized_tensor}}, {output_layer}, {},
                            &outputs);
  inference_timer.Stop();
  cout << "Inference Time in ms: " << inference_timer.ElapsedInMS() << endl;

  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }

  // This is for automated testing to make sure we get the expected result with
  // the default settings. We know that label 653 (military uniform) should be
  // the top label for the Admiral Hopper image.
  if (self_test) {
    bool expected_matches;
    Status check_status = CheckTopLabel(outputs, 653, &expected_matches);
    if (!check_status.ok()) {
      LOG(ERROR) << "Running check failed: " << check_status;
      return -1;
    }
    if (!expected_matches) {
      LOG(ERROR) << "Self-test failed!";
      return -1;
    }
  }

  // Do something interesting with the results we've generated.
  string labels = tensorflow::io::JoinPath(root_dir, labels_file);
  Status print_status = PrintTopLabels(outputs, labels);
  if (!print_status.ok()) {
    LOG(ERROR) << "Running print failed: " << print_status;
    return -1;
  }

  return 0;
}
