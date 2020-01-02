/*******************************************************************************
 * Copyright 2017-2019 Intel Corporation
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

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

#include "ngraph_bridge/ngraph_api.h"
#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/version.h"

using namespace std;

// Prints the available backends
int PrintAvailableBackends() {
  // Get the list of backends
  auto supported_backends =
      tensorflow::ngraph_bridge::BackendManager::GetSupportedBackendNames();
  vector<string> backends(supported_backends.begin(), supported_backends.end());
  if (backends.empty()) {
    std::cout << "No backend available " << std::endl;
    return -1;
  }
  std::cout << "Available backends: " << std::endl;
  for (auto& backend_name : backends) {
    std::cout << "Backend: " << backend_name << std::endl;
  }
  return 0;
}

// Create a simple computation graph and run
void RunSimpleNetworkExample() {
  // Create the graph
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  // Matrix A = [3 2; -1 0]
  auto A = tensorflow::ops::Const(root, {{0.03f, 0.022f}, {-0.001f, 0.025f}});
  // Vector b = [3 5]
  auto b = tensorflow::ops::Const(root, {{0.345f, 0.35f}});
  // v = Ab^T
  auto v = tensorflow::ops::MatMul(root.WithOpName("v"), A, b,
                                   tensorflow::ops::MatMul::TransposeB(true));
  // R = softmax(v)
  auto R = tensorflow::ops::Softmax(root, v);

  // Turn off optimizations so that all the nodes are processed
  tensorflow::SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(tensorflow::OptimizerOptions_Level_L0);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(tensorflow::RewriterConfig::OFF);

  if (tensorflow::ngraph_bridge::ngraph_tf_is_grappler_enabled()) {
    auto* custom_config = options.config.mutable_graph_options()
                              ->mutable_rewrite_options()
                              ->add_custom_optimizers();

    custom_config->set_name("ngraph-optimizer");
    (*custom_config->mutable_parameter_map())["ngraph_backend"].set_s("CPU");
    (*custom_config->mutable_parameter_map())["device_id"].set_s("1");

    options.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_min_graph_nodes(-1);

    options.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_meta_optimizer_iterations(tensorflow::RewriterConfig::ONE);
  }

  tensorflow::ClientSession session(root, options);

  std::vector<tensorflow::Tensor> outputs;
  session.Run({R}, &outputs);

  // Print the output
  std::cout << "Result: " << outputs[0].matrix<float>() << std::endl;
}

void PrintVersion() {
  // nGraph Bridge version info
  std::cout << "Bridge version: "
            << tensorflow::ngraph_bridge::ngraph_tf_version() << std::endl;
  std::cout << "nGraph version: "
            << tensorflow::ngraph_bridge::ngraph_lib_version() << std::endl;
  std::cout << "CXX11_ABI Used: "
            << tensorflow::ngraph_bridge::ngraph_tf_cxx11_abi_flag()
            << std::endl;
  std::cout << "Grappler Enabled? "
            << (tensorflow::ngraph_bridge::ngraph_tf_is_grappler_enabled()
                    ? std::string("Yes")
                    : std::string("No"))
            << std::endl;
  std::cout << "Variables Enabled? "
            << (tensorflow::ngraph_bridge::ngraph_tf_are_variables_enabled()
                    ? std::string("Yes")
                    : std::string("No"))
            << std::endl;
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  PrintVersion();

  // Print the avialable backends and if none are available
  // error out
  if (PrintAvailableBackends()) {
    return -1;
  }

  // Run a simple example
  RunSimpleNetworkExample();

  return 0;
}
