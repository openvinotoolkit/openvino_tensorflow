/*******************************************************************************
 * Copyright 2019-2020 Intel Corporation
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

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"

#include "ngraph_bridge/version.h"

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

namespace benchmark {

InferenceEngine::InferenceEngine(const string& name) : m_name(name) {}

Status InferenceEngine::LoadImage(const string& network,
                                  const std::vector<string>& image_files,
                                  int input_width, int input_height,
                                  float input_mean, float input_std,
                                  const string& input_layer,
                                  const string& output_layer, bool use_NCHW,
                                  bool preload_images, int input_channels) {
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
    std::vector<tf::Tensor> resized_tensors;
    TF_CHECK_OK(ReadTensorFromImageFile(
        m_image_files, m_input_height, m_input_width, m_input_mean, m_input_std,
        m_use_NCHW, m_input_channels, &resized_tensors));
    m_image_to_repeat = resized_tensors[0];
  }
  // Now compile the graph if needed
  // This would be useful to detect errors early. For a graph
  // that has already undergone TensorFlow to nGraph (may be via tf2ngraph.py)
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

Status InferenceEngine::CreateSession(const string& graph_filename,
                                      unique_ptr<Session>& session) {
  SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions_Level_L0);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(RewriterConfig::OFF);
  options.config.set_inter_op_parallelism_threads(2);

  // The following is related to Grappler - which we are turning off
  // Until we get a library fully running
  if (tf::ngraph_bridge::is_grappler_enabled()) {
    auto* custom_config = options.config.mutable_graph_options()
                              ->mutable_rewrite_options()
                              ->add_custom_optimizers();

    custom_config->set_name("ngraph-optimizer");
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

}  // namespace benchmark
