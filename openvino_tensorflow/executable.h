/*****************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
*****************************************************************************/

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"

#include "openvino_tensorflow/ie_backend_engine.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

// A Inference Engine executable object produced by compiling an
// OpenVINO Model.
class Executable {
 public:
  Executable(shared_ptr<ov::Model> model, string device, string device_type);
  ~Executable() {}
  bool Call(const vector<shared_ptr<ov::Tensor>>& inputs,
            vector<shared_ptr<ov::Tensor>>& outputs,
            bool multi_req_execution = false);

  const ov::ResultVector& GetResults() { return m_model->get_results(); };

  const vector<size_t> GetOutputShape(const int i) {
    if (m_trivial_fn) {
      return GetResults()[i]->get_shape();
    } else {
      return m_ie_engine->get_output_shape(i);
    }
  }

  void SetOutputShapes(vector<ov::Shape> ng_output_shapes) {
    m_ng_output_shapes = ng_output_shapes;
  }

  const vector<ov::Shape> GetOutputShapes() { return m_ng_output_shapes; }

  void ExportIR(const string& output_dir);

 private:
  bool CallTrivial(const vector<shared_ptr<ov::Tensor>>& inputs,
                   vector<shared_ptr<ov::Tensor>>& outputs);

  string m_device;
  string m_device_type;
  // This holds the parameters we insert for functions with no input parameters
  vector<pair<string, shared_ptr<ov::Tensor>>> m_hoisted_params;
  vector<int> m_skipped_inputs;
  vector<ov::Shape> m_ng_output_shapes;
  // This keeps track of whether the original function was trivial: either a
  // constant function, an identity function or a zero function
  shared_ptr<ov::Model> m_trivial_fn;
  // This is the original OpenVINO model corresponding to this executable
  shared_ptr<ov::Model> m_model;
  shared_ptr<IE_Backend_Engine> m_ie_engine;
  std::vector<std::string> m_in_names;
};
}  // namespace openvino_tensorflow
}  // namespace tensorflow
