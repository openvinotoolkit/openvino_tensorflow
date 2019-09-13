/*******************************************************************************
 * Copyright 2019 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include <thread>
#include "vector"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "ngraph/event_tracing.hpp"

#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/version.h"

using namespace std;
namespace tf = tensorflow;

extern tf::Status LoadGraph(const string& graph_file_name,
                            std::unique_ptr<tf::Session>* session,
                            const tf::SessionOptions& options);

extern tf::Status ReadTensorFromImageFile(const std::vector<string>& file_name,
                                          const int input_height,
                                          const int input_width,
                                          const float input_mean,
                                          const float input_std, bool use_NCHW,
                                          const int input_channels,
                                          std::vector<tf::Tensor>* out_tensors);

extern tf::Status PrintTopLabels(const std::vector<tf::Tensor>& outputs,
                                 const string& labels_file_name);

// Prints the available backends
void PrintAvailableBackends() {
  // Get the list of backends
  auto supported_backends =
      tf::ngraph_bridge::BackendManager::GetSupportedBackendNames();
  vector<string> backends(supported_backends.begin(), supported_backends.end());

  cout << "Available backends: " << endl;
  for (auto& backend_name : backends) {
    cout << "Backend: " << backend_name << std::endl;
  }
}

// Sets the specified backend. This backend must be set BEFORE running
// the computation
tf::Status SetNGraphBackend(const string& backend_name) {
  // Select a backend
  tf::Status status =
      tf::ngraph_bridge::BackendManager::SetBackendName(backend_name);
  return status;
}

void PrintVersion() {
  // nGraph Bridge version info
  std::cout << "Bridge version: " << tf::ngraph_bridge::ngraph_tf_version()
            << std::endl;
  std::cout << "nGraph version: " << tf::ngraph_bridge::ngraph_lib_version()
            << std::endl;
  std::cout << "CXX11_ABI Used: "
            << tf::ngraph_bridge::ngraph_tf_cxx11_abi_flag() << std::endl;
  std::cout << "Grappler Enabled? "
            << (tf::ngraph_bridge::ngraph_tf_is_grappler_enabled()
                    ? std::string("Yes")
                    : std::string("No"))
            << std::endl;
  std::cout << "Variables Enabled? "
            << (tf::ngraph_bridge::ngraph_tf_are_variables_enabled()
                    ? std::string("Yes")
                    : std::string("No"))
            << std::endl;

  PrintAvailableBackends();
}

std::unique_ptr<tf::Session> CreateSession(const string& graph_filename) {
  tf::SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(tf::OptimizerOptions_Level_L0);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(tf::RewriterConfig::OFF);

  // The following is related to Grapller - which we are turning off
  // Until we get a library fully running
  if (tf::ngraph_bridge::ngraph_tf_is_grappler_enabled()) {
    options.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->add_custom_optimizers()
        ->set_name("ngraph-optimizer");

    options.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_min_graph_nodes(-1);

    options.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_meta_optimizer_iterations(tf::RewriterConfig::ONE);
  }

  // Load the network
  std::unique_ptr<tf::Session> session;
  tf::Status load_graph_status = LoadGraph(graph_filename, &session, options);

  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return nullptr;
  }
  return std::move(session);
}

int main(int argc, char** argv) {
  // parameters below need to modified as per model
  string image = "image_00000.png";
  int batch_size = 1;
  // Vector size is same as the batch size, populating with single image
  std::vector<string> images(batch_size, image);
  string graph =
      "resnet50_nchw_optimized_frozen_resnet_v1_50_nchw_cifar_fullytrained_"
      "fullyquantized_02122019.pb";
  string labels = "";
  int input_width = 224;
  int input_height = 224;
  float input_mean = 128.0;
  float input_std = 1;
  string input_layer = "input";
  string output_layer = "resnet_v1_50/predictions/Softmax";
  bool use_NCHW = true;
  int input_channels = 3;

  std::vector<tf::Flag> flag_list = {
      tf::Flag("image", &image, "image to be processed"),
      tf::Flag("graph", &graph, "graph to be executed"),
      tf::Flag("labels", &labels, "name of file containing labels"),
      tf::Flag("input_width", &input_width,
               "resize image to this width in pixels"),
      tf::Flag("input_height", &input_height,
               "resize image to this height in pixels"),
      tf::Flag("input_mean", &input_mean, "scale pixel values to this mean"),
      tf::Flag("input_std", &input_std,
               "scale pixel values to this std deviation"),
      tf::Flag("input_layer", &input_layer, "name of input layer"),
      tf::Flag("output_layer", &output_layer, "name of output layer"),
      tf::Flag("use_NCHW", &use_NCHW, "Input data in NCHW format"),
  };

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

  const char* backend = "CPU";
  int num_images_for_each_thread = 10;

  if (argc > 1) {
    num_images_for_each_thread = atoi(argv[1]);
  }

  if (SetNGraphBackend(backend) != tf::Status::OK()) {
    std::cout << "Error: Cannot set the backend: " << backend << std::endl;
    return -1;
  }

  std::cout << "Component versions\n";
  PrintVersion();

  std::cout << "\nCreating session\n";
  ngraph::Event session_create_event("Session Create", "", "");

  // Run the MatMul example
  // auto session = CreateSession("inception_v3_2016_08_28_frozen.pb");
  auto session = CreateSession(graph);
  session_create_event.Stop();
  ngraph::Event::write_trace(session_create_event);

  // Create threads and fire up the images
  const int NUM_THREADS = 2;
  std::thread threads[NUM_THREADS];

  std::cout << "Running inferences\n";

  for (int i = 0; i < NUM_THREADS; i++) {
    threads[i] = std::thread([=, &session] {
      for (int iter_count = 0; iter_count < num_images_for_each_thread;
           iter_count++) {
        std::ostringstream oss;
        oss << "Read(" << i << ") [" << iter_count << "]";
        ngraph::Event read_event(oss.str(), "Image reading", "");

        // Read image
        std::vector<tf::Tensor> resized_tensors;
        tf::Status read_tensor_status = ReadTensorFromImageFile(
            images, input_height, input_width, input_mean, input_std, use_NCHW,
            input_channels, &resized_tensors);

        if (!read_tensor_status.ok()) {
          LOG(ERROR) << read_tensor_status;
          continue;
        }
        read_event.Stop();

        // Run inference
        oss.clear();
        oss.seekp(0);
        oss << "Infer(" << i << ") [" << iter_count << "]";
        ngraph::Event infer_event(oss.str(), "Inference", "");

        const tf::Tensor& resized_tensor = resized_tensors[0];
        string input_layer = "input";
        std::vector<tf::Tensor> outputs;

        tf::Status run_status = session->Run({{input_layer, resized_tensor}},
                                             {output_layer}, {}, &outputs);
        if (!run_status.ok()) {
          LOG(ERROR) << "Running model failed: " << run_status;
        }
        infer_event.Stop();

        // Write the events
        ngraph::Event::write_trace(read_event);
        ngraph::Event::write_trace(infer_event);
      }
    });
  }

  // Wait until everyone is done
  for (auto& next_thread : threads) {
    next_thread.join();
  }

  std::cout << "Done\n";
  return 0;
}
