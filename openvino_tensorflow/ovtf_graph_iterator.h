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
#include "openvino_tensorflow/ovtf_decoder.h"

#include "tensorflow/core/graph/graph.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

// A Inference Engine executable object produced by compiling an nGraph
// function.
class OVTFGraphIterator : public ov::frontend::tensorflow::GraphIterator {
 public:
  OVTFGraphIterator(const std::vector<::tensorflow::Node*> nodes) {
    m_nodes.resize(nodes.size());
    for (size_t i = 0; i < m_nodes.size(); ++i) m_nodes[i] = &(nodes[i]->def());
  }

  /// Set iterator to the start position
  void reset() override { m_node_index = 0; }

  size_t size() const override { return m_nodes.size(); }

  /// Moves to the next node in the graph
  void next() override { m_node_index++; }

  bool is_end() const override { return m_node_index >= m_nodes.size(); }

  /// Return NodeContext for the current node that iterator points to
  std::shared_ptr<ov::frontend::tensorflow::DecoderBase> get_decoder()
      const override {
    return std::make_shared<OVTFDecoder>(m_nodes[m_node_index]);
  }

  void sort_nodes(std::vector<std::string>& ordered_names) {
    int ordered_idx = 0;
    for (int i = 0; i < ordered_names.size(); i++) {
      for (int j = ordered_idx; j < m_nodes.size(); j++) {
        if (ordered_names[i] == m_nodes[j]->name()) {
          const ::tensorflow::NodeDef* current = m_nodes[ordered_idx];
          m_nodes[ordered_idx] = m_nodes[j];
          m_nodes[j] = current;
          ordered_idx++;
          break;
        }
      }
    }
  }

 private:
  std::vector<const ::tensorflow::NodeDef*> m_nodes;
  size_t m_node_index = 0;
};
}  // namespace openvino_tensorflow
}  // namespace tensorflow
