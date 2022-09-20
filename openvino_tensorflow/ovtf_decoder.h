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
#include "openvino/frontend/tensorflow/decoder.hpp"
#include "openvino/frontend/tensorflow/graph_iterator.hpp"

#include "tensorflow/core/graph/graph.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

// A Inference Engine executable object produced by compiling an nGraph
// function.
class OVTFDecoder : public ov::frontend::tensorflow::DecoderBase {
 public:
  explicit OVTFDecoder(const ::tensorflow::NodeDef* node_def)
      : m_node_def(node_def) {}

  ov::Any get_attribute(const std::string& name) const override;

  size_t get_input_size() const override;

  void get_input_node(const size_t input_port_idx, std::string& producer_name,
                      size_t& producer_output_port_index) const override;

  const std::string& get_op_type() const override;

  const std::string& get_op_name() const override;

 private:
  vector<::tensorflow::AttrValue> decode_attribute_helper(
      const string& name) const;
  const ::tensorflow::NodeDef* m_node_def;
};
}  // namespace openvino_tensorflow
}  // namespace tensorflow
