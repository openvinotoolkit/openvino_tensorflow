/*****************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
*****************************************************************************/

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <ie_core.hpp>
#include "ngraph/ngraph.hpp"

#include "openvino_tensorflow/ie_backend_engine.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

// A Inference Engine executable object produced by compiling an nGraph
// function.
class Executable {
 public:
  Executable(shared_ptr<ngraph::Function> func, string device,
             string device_type);
  ~Executable() {}
  bool Call(const vector<shared_ptr<ngraph::runtime::Tensor>>& inputs,
            vector<shared_ptr<ngraph::runtime::Tensor>>& outputs,
            bool multi_req_execution = false);

  const ngraph::ResultVector& GetResults() {
    return m_function->get_results();
  };

  const vector<size_t> GetOutputShape(const int i) {
    if (m_trivial_fn) {
      return GetResults()[i]->get_shape();
    } else {
      return m_ie_engine->get_output_shape(i);
    }
  }

  void ExportIR(const string& output_dir);

 private:
  bool CallTrivial(const vector<shared_ptr<ngraph::runtime::Tensor>>& inputs,
                   vector<shared_ptr<ngraph::runtime::Tensor>>& outputs);

  InferenceEngine::CNNNetwork m_network;
  InferenceEngine::InferRequest m_infer_req;
  string m_device;
  string m_device_type;
  // This holds the parameters we insert for functions with no input parameters
  vector<pair<string, shared_ptr<ngraph::runtime::Tensor>>> m_hoisted_params;
  vector<int> m_skipped_inputs;
  // This keeps track of whether the original function was trivial: either a
  // constant function, an identity function or a zero function
  shared_ptr<ngraph::Function> m_trivial_fn;
  // This is the original nGraph function corresponding to this executable
  shared_ptr<ngraph::Function> m_function;
  shared_ptr<IE_Backend_Engine> m_ie_engine;
};
}  // namespace openvino_tensorflow
}  // namespace tensorflow
