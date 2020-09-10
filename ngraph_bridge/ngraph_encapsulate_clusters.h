/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
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

#ifndef NGRAPH_TF_BRIDGE_ENCAPSULATE_CLUSTERS_H_
#define NGRAPH_TF_BRIDGE_ENCAPSULATE_CLUSTERS_H_
#pragma once

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ngraph/ngraph.hpp"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
namespace ngraph_bridge {

// TODO: an optimization would be to separate the analysis and rewriting passes
// cleanly, so that analysis pass is run in mark_for_clustering, and its
// information is reused here instead of recalculating
// To do that an Encapsulator object with AnalysisPass run can be created in
// MarkForClustering, and that can be passed to EncapsulateClusters

/// Takes a TF graph where ngraph_cluster attributes has been marked in a
/// preceeding pass (assign_clusters), then replaces TF subgraphs and inserts
/// encapsulate ops in their place.
Status EncapsulateClusters(
    Graph* graph, int graph_id,
    const std::unordered_map<std::string, std::string>& device_config);

// TODO Encapsulator is dependent on ClusterManager. They could be made
// independent.

// A class to perform analysis (identify subgraphs)
// and rewriting (create encapsulates and splice them in)
// Order of calling: construction -> AnalysisPass -> RewritePass
//                                       |
//                                       v
//                                 NewClusterIds
// Any other order of calling will generate errors
// Cannot be copied/moved or reset
class Encapsulator {
 public:
  Encapsulator(Graph* g);
  // Populate ClusterManager with the subgraphs for each potential encapsulate
  Status AnalysisPass();
  // Perform the actual graph surgery
  Status RewritePass(
      int graph_id,
      const std::unordered_map<std::string, std::string>& device_config);
  // Returns the newly created cluster ids after AnalysisPass is done
  // Needed because ClusterManager (CM) might have contained old stuff,
  // so it might not be possible to query the CM itself to get this
  Status GetNewClusterIDs(std::set<int>& result);

  Encapsulator(const Encapsulator&) = delete;
  Encapsulator(Encapsulator&&) = delete;
  Encapsulator& operator=(const Encapsulator&) = delete;
  Encapsulator& operator=(Encapsulator&&) = delete;

 private:
  Graph* graph;
  // boolean to indicate if analysis has been done
  // If not rewritepass should not be called
  bool analysis_done;
  // boolean to indicate that rewrite is done;
  bool rewrite_done;
  // A map from cluster indices to the expected device name for nodes
  // in that cluster.
  std::map<int, std::string> device_name_map;

  // As we build the graph we will be tracking the.. TODO(amprocte): finish
  // this comment.
  std::map<std::tuple<int, int>, std::tuple<int, int>> output_remap_map;
  std::map<std::tuple<int, int, int>, int> input_remap_map;
  std::map<std::tuple<int, std::string, int>, string> input_rename_map;

  // A map from cluster indices to a vector of input data types.
  std::map<int, std::vector<std::tuple<int, int, DataType>>> cluster_input_map;
  // A map from cluster indices to a vector of output data types.
  std::map<int, std::vector<DataType>> cluster_output_dt_map;

  // A map from cluster indices to corresponding NGraphEncapsulate nodes.
  std::map<int, Node*> cluster_node_map;

  std::set<int> cluster_indices_for_this_graph;

  static void AddInput(NodeDef* dst, StringPiece src_name, int src_slot);
};

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif  // NGRAPH_TF_BRIDGE_ENCAPSULATE_CLUSTERS_H_
