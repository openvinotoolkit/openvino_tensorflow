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
  OVTFGraphIterator(const ::tensorflow::GraphDef* graph_def)
      : m_graph_def(graph_def) {
    m_nodes.resize(m_graph_def->node_size());
    for (size_t i = 0; i < m_nodes.size(); ++i)
      m_nodes[i] = &m_graph_def->node(i);
  }

  /// Set iterator to the start position
  void reset() override { node_index = 0; }

  size_t size() const override { return m_nodes.size(); }

  /// Moves to the next node in the graph
  void next() override { node_index++; }

  bool is_end() const override { return node_index >= m_nodes.size(); }

  /// Return NodeContext for the current node that iterator points to
  std::shared_ptr<ov::frontend::tensorflow::DecoderBase> get_decoder()
      const override {
    return std::make_shared<OVTFDecoder>(m_nodes[node_index]);
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
  size_t node_index = 0;
  const ::tensorflow::GraphDef* m_graph_def;
};
}  // namespace openvino_tensorflow
}  // namespace tensorflow
