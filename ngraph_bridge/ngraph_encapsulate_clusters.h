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

#ifndef NGRAPH_TF_BRIDGE_ENCAPSULATE_CLUSTERS_H_
#define NGRAPH_TF_BRIDGE_ENCAPSULATE_CLUSTERS_H_
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include <iostream>
#include "tensorflow/core/graph/graph.h"

#include "ngraph/ngraph.hpp"

#include "ngraph_bridge/ngraph_partial_shapes.h"

namespace tensorflow {

namespace ngraph_bridge {

typedef std::map<std::string, std::vector<int>> ShapeHintMap;

// the integer represent AOT level requested.
typedef std::pair<bool, std::set<ShapeHintMap>> AOTInfo;

// TODO: an optimization would be to separate the analysis and rewriting passes
// cleanly, so that analysis pass is run in mark_for_clustering, and its
// information is reused here instead of recalculating
// To do that an Encapsulator object with AnalysisPass run can be created in
// MarkForClustering, and that can be passed to EncapsulateClusters

/// Takes a TF graph where ngraph_cluster attributes has been marked in a
/// preceeding pass (assign_clusters), then replaces TF subgraphs and inserts
/// encapsulate ops in their place. Optionally can perform ahead of time
/// compilation.
Status EncapsulateClusters(
    Graph* graph, int graph_id, FunctionDefLibrary* fdeflib,
    const std::unordered_map<std::string, std::string>& device_config,
    const AOTInfo& aot_info);

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
      FunctionDefLibrary* fdeflib, int graph_id,
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

  // We *should* eventually have a way of monitoring the device and the backend
  // together
  std::map<int, std::string> backend_name_map;

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

// Translates TF subgraph to ng function then compiles it
Status PerformAOTOnEncapsulates(Graph* graph, const AOTInfo& aot_info);

std::string HintAsString(ShapeHintMap single_hint);

// Given a node, partialshape info from TF (present in the .pb itself) and a
// shape hint, combine all that information
PartialShape CombineNodeInfoAndHint(Node* node,
                                    PartialShape partial_shape_from_node,
                                    const ShapeHintMap& single_hint);

// Given a TF graph, it scans it for inputs and finds what TF is saying about
// their shapes (in the .pb itself)
// Creates a map between input node names and PartialShape information we get
// from the TF graph
std::map<std::string, PartialShape> GetShapesFromTFInputnodes(
    Graph* graph, const string& input_node_type);

// Given an encapsulate node, and the input shapes,
// performs TranslateGraph and returns an ng function and a signature
Status PerformTranslation(Node* node,
                          const std::map<std::string, std::vector<int>>&
                              inputs_node_shapes_for_compilation,
                          std::string& signature,
                          std::shared_ptr<ngraph::Function>& ng_function);

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif  // NGRAPH_TF_BRIDGE_ENCAPSULATE_CLUSTERS_H_
