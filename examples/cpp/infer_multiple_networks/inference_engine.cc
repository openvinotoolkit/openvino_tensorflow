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

#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>

#include "inference_engine.h"
#include "ngraph/event_tracing.hpp"
#include "ngraph_backend_manager.h"
#include "version.h"

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

using tensorflow::SessionOptions;
using tensorflow::RewriterConfig;
using tensorflow::OptimizerOptions_Level_L0;
using tensorflow::Tensor;
using std::cout;
using std::move;
using std::ostringstream;
namespace tf = tensorflow;

extern tf::Status LoadGraph(const string& graph_file_name,
                            std::unique_ptr<tf::Session>* session,
                            const tf::SessionOptions& options);

extern tf::Status ReadTensorFromImageFile(const std::vector<string>& file_names,
                                          const int input_height,
                                          const int input_width,
                                          const float input_mean,
                                          const float input_std, bool use_NCHW,
                                          const int input_channels,
                                          std::vector<tf::Tensor>* out_tensors);

namespace infer_multiple_networks {

InferenceEngine::InferenceEngine(const string& name, const string& backend)
    : m_name(name) {}
Status InferenceEngine::Load(const string& network,
                             const std::vector<string>& image_files,
                             int input_width, int input_height,
                             float input_mean, float input_std,
                             const string& input_layer,
                             const string& output_layer, bool use_NCHW,
                             bool preload_images, int input_channels) {
  // Load the network
  TF_CHECK_OK(CreateSession(network, m_session));

  // Save the input related information
  m_image_files = image_files;
  m_input_width = input_width;
  m_input_height = input_height;
  m_input_mean = input_mean;
  m_input_std = input_std;
  m_input_layer = input_layer;
  m_output_layer = output_layer;
  m_use_NCHW = use_NCHW;
  m_preload_images = preload_images;
  m_input_channels = input_channels;

  // Preload the image is requested
  if (m_preload_images) {
    // Set the CPU as the backend before these ops
    string current_backend;
    TF_CHECK_OK(tf::ngraph_bridge::BackendManager::GetCurrentlySetBackendName(
        &current_backend));
    TF_CHECK_OK(tf::ngraph_bridge::BackendManager::SetBackendName("CPU"));
    std::vector<tf::Tensor> resized_tensors;
    TF_CHECK_OK(ReadTensorFromImageFile(
        m_image_files, m_input_height, m_input_width, m_input_mean, m_input_std,
        m_use_NCHW, m_input_channels, &resized_tensors));
    m_image_to_repeat = resized_tensors[0];
    TF_CHECK_OK(
        tf::ngraph_bridge::BackendManager::SetBackendName(current_backend));
  }
  // Now compile the graph if needed
  // This would be useful to detect errors early. For a graph
  // that had already undergone TensorFlow to nGraph (may be via tf2ngraph.py)
  // won't need any compilation though as that graph will most likely have
  // the executable available as well
  // TODO

  return Status::OK();
}

InferenceEngine::~InferenceEngine() {
  if (m_worker.joinable()) {
    m_worker.join();
  }
}

Status InferenceEngine::Start(const function<void(int)>& step_callback) {
  m_step_callback = step_callback;
  return Start();
}

Status InferenceEngine::Start() {
  thread new_worker(&InferenceEngine::ThreadMain, this);
  m_worker = move(new_worker);
  return Status::OK();
}

void InferenceEngine::ThreadMain() {
  m_terminate_worker = false;
  int step_count = 0;

  while (true) {
    ostringstream ss;
    ss << "[" << m_name << "] Iteration: " << step_count;
    ngraph::Event itreation_event(ss.str(), "", "");

    if (!m_preload_images) {
      // Read the image
      cout << "[" << m_name << "] " << step_count << ": Reading image\n";
      ngraph::Event read_event("Read", "", "");

      string current_backend;
      TF_CHECK_OK(tf::ngraph_bridge::BackendManager::GetCurrentlySetBackendName(
          &current_backend));
      TF_CHECK_OK(tf::ngraph_bridge::BackendManager::SetBackendName("CPU"));

      std::vector<tf::Tensor> resized_tensors;
      TF_CHECK_OK(ReadTensorFromImageFile(
          m_image_files, m_input_height, m_input_width, m_input_mean,
          m_input_std, m_use_NCHW, m_input_channels, &resized_tensors));

      m_image_to_repeat = resized_tensors[0];
      TF_CHECK_OK(
          tf::ngraph_bridge::BackendManager::SetBackendName(current_backend));

      read_event.Stop();
      ngraph::Event::write_trace(read_event);
    }

    // Submit for inference
    cout << "[" << m_name << "] " << step_count
         << ": Submit image for inference\n";
    ngraph::Event infer_event("Infer", "", "");

    const tf::Tensor& resized_tensor = m_image_to_repeat;
    std::vector<Tensor> outputs;
    TF_CHECK_OK(m_session->Run({{m_input_layer, resized_tensor}},
                               {m_output_layer}, {}, &outputs));
    infer_event.Stop();

    ngraph::Event::write_trace(infer_event);

    if (m_step_callback != nullptr) {
      m_step_callback(step_count);
    }
    step_count++;

    itreation_event.Stop();
    ngraph::Event::write_trace(itreation_event);

    // Check if we are asked to terminate
    if (m_terminate_worker) {
      cout << "[" << m_name << "] m_terminate_worker: Signaled" << std::endl;
      break;
    }
  }
  cout << "[" << m_name << "] Worker terminating\n";
}

Status InferenceEngine::Stop() {
  cout << "[" << m_name << "] Stop called" << std::endl;
  m_terminate_worker = true;
  return Status::OK();
}

Status InferenceEngine::CreateSession(const string& graph_filename,
                                      unique_ptr<Session>& session) {
  SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions_Level_L0);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(RewriterConfig::OFF);

  // The following is related to Grappler - which we are turning off
  // Until we get a library fully running
  if (tf::ngraph_bridge::ngraph_tf_is_grappler_enabled()) {
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

  // Load the network
  Status load_graph_status = LoadGraph(graph_filename, &session, options);
  return load_graph_status;
}

}  // namespace infer_multiple_networks
