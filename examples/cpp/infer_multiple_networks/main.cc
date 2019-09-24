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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "ngraph/event_tracing.hpp"

#include "examples/cpp/infer_multiple_networks/inference_engine.h"
#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/version.h"

using namespace std;
namespace tf = tensorflow;

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

int main(int argc, char** argv) {
  // parameters below need to modified as per model
  string image = "grace_hopper.jpg";
  int batch_size = 1;
  // Vector size is same as the batch size, populating with single image
  vector<string> images(batch_size, image);
  string graph = "inception_v3_2016_08_28_frozen.pb";
  string labels = "";
  int input_width = 299;
  int input_height = 299;
  float input_mean = 0.0;
  float input_std = 255;
  string input_layer = "input";
  string output_layer = "InceptionV3/Predictions/Reshape_1";
  bool use_NCHW = false;
  bool preload_images = true;
  int input_channels = 3;
  int iteration_count = 10;

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
      tf::Flag("iteration_count", &iteration_count,
               "How many times to repeat the inference"),
      tf::Flag("preload_images", &preload_images,
               "Repeat the same image for inference"),
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
  if (SetNGraphBackend(backend) != tf::Status::OK()) {
    std::cout << "Error: Cannot set the backend: " << backend << std::endl;
    return -1;
  }

  std::cout << "Component versions\n";
  PrintVersion();

  infer_multiple_networks::InferenceEngine infer_engine_1("engine_1");
  TF_CHECK_OK(infer_engine_1.Load(
      graph, images, input_width, input_height, input_mean, input_std,
      input_layer, output_layer, use_NCHW, preload_images, input_channels));
  infer_multiple_networks::InferenceEngine infer_engine_2("engine_2");
  TF_CHECK_OK(infer_engine_2.Load(
      graph, images, input_width, input_height, input_mean, input_std,
      input_layer, output_layer, use_NCHW, preload_images, input_channels));
  infer_multiple_networks::InferenceEngine infer_engine_3("engine_3");
  TF_CHECK_OK(infer_engine_3.Load(
      graph, images, input_width, input_height, input_mean, input_std,
      input_layer, output_layer, use_NCHW, preload_images, input_channels));

  bool engine_1_running = true;
  TF_CHECK_OK(infer_engine_1.Start([&](int step_count) {
    if (step_count == (iteration_count - 1)) {
      TF_CHECK_OK(infer_engine_1.Stop());
      engine_1_running = false;
    }
  }));

  bool engine_2_running = true;
  TF_CHECK_OK(infer_engine_2.Start([&](int step_count) {
    if (step_count == (iteration_count - 1)) {
      TF_CHECK_OK(infer_engine_2.Stop());
      engine_2_running = false;
    }
  }));

  bool engine_3_running = true;
  TF_CHECK_OK(infer_engine_3.Start([&](int step_count) {
    if (step_count == (iteration_count - 1)) {
      TF_CHECK_OK(infer_engine_3.Stop());
      engine_3_running = false;
    }
  }));

  while (engine_1_running || engine_2_running || engine_3_running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }

  std::cout << "Done" << std::endl;
  return 0;
}
