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

#include "ngraph_optimizer.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"

#include <iomanip>
#if defined NGRAPH_DISTRIBUTED
#include "ngraph/distributed.hpp"
#endif

#include <iostream>

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {

Status NgraphOptimizer::Optimize(tensorflow::grappler::Cluster* cluster,
                                 const tensorflow::grappler::GrapplerItem& item,
                                 GraphDef* output) {
  NGRAPH_VLOG(0) << "[NGTF-OPTIMIZER] Here at NgraphOptimizer ";
  NGRAPH_VLOG(5) << "[NGTF-OPTIMIZER] grappler item id " << item.id;

  // Convert the GraphDef to Graph
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.expect_device_spec = true;
  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, item.graph, &graph));

  // For filename generation purposes, grab a fresh index. This is just an
  // arbitrary integer to avoid filename collisions resulting from subsequent
  // runs of this pass.
  int idx = FreshIndex();

  // If requested, dump pre-capture graphs.
  if (DumpPrecaptureGraphs()) {
    DumpGraphs(graph, idx, "precapture", "Pre-Capture Graph");
  }

  // If ngraph is disabled via ngraph_bridge api or NGRAPH_TF_DISABLE is set
  // we will not do anything; all subsequent
  // passes become a no-op.
  if (config::IsEnabled() == false ||
      std::getenv("NGRAPH_TF_DISABLE") != nullptr) {
    NGRAPH_VLOG(0) << "[NGTF-OPTIMIZER] Ngraph is disabled ";
    graph.ToGraphDef(output);
    return Status::OK();
  }

  // Get the nodes to be skipped
  std::set<string> fetch_nodes;
  for (const string& f : item.fetch) {
    int pos = f.find(":");
    fetch_nodes.insert(f.substr(0, pos));
  }
  std::set<string>& skip_these_nodes = fetch_nodes;

  // TODO [kkhanna] : Move adding IdentityN to a separate file
  // Rewrite graph to add IdentityN node so the fetch node can be encapsulated
  // as well
  // If the fetch node in question has 0 outputs or any of the outputs
  // has ref type as a data type then don't add IdentityN node, but the fetch
  // node will be skipped from capturing and marking for clustering.
  Graph* input_graph = &graph;
  for (auto node : input_graph->op_nodes()) {
    bool fetch_node = false;
    bool ref_type = false;
    fetch_node = fetch_nodes.find(node->name()) != fetch_nodes.end();
    if (fetch_node) {
      NGRAPH_VLOG(5) << "[NGTF-OPTIMIZER] Fetch Node " << node->name();
      // Check the number of outputs of the 'fetch_node'
      // Only move further to create an IdentityN node
      // if it is greater than 0
      // Also, make sure that none of the output types is
      // a ref type because IdentityN does not support
      // an input of type ref type
      if (node->num_outputs()) {
        std::vector<NodeBuilder::NodeOut> inputs;
        std::vector<DataType> input_types;
        for (int i = 0; i < node->num_outputs(); i++) {
          if (IsRefType(node->output_type(i))) {
            NGRAPH_VLOG(5) << "[NGTF-OPTIMIZER] "
                           << "Datatype for the node output"
                           << " at index " << i << "is ref type";
            ref_type = true;
            break;
          }
          input_types.push_back(node->output_type(i));
          inputs.push_back(NodeBuilder::NodeOut(node, i));
        }

        if (ref_type) {
          NGRAPH_VLOG(5)
              << "[NGTF-OPTIMIZER] Cannot construct an IdentityN node";
          continue;
        }

        NGRAPH_VLOG(5) << "[NGTF-OPTIMIZER] Creating an IdentityN node";
        Node* identityN_node;
        TF_RETURN_IF_ERROR(NodeBuilder(node->name(), "IdentityN")
                               .Attr("T", input_types)
                               .Input(inputs)
                               .Device(node->assigned_device_name())
                               .Finalize(input_graph, &identityN_node));

        identityN_node->set_assigned_device_name(node->assigned_device_name());

        // Rename the skip node
        // Get a new name for the node with the given prefix
        // We will use the 'original-node-name_ngraph' as the prefix
        string new_name = input_graph->NewName(node->name() + "_ngraph");
        // TODO: Use (guaranteed) unique name here
        node->set_name(new_name);
        NGRAPH_VLOG(5) << "[NGTF-OPTIMIZER] New name for fetch node "
                       << node->name();
      } else {
        NGRAPH_VLOG(5) << "[NGTF-OPTIMIZER] num outputs "
                       << node->num_outputs();
        NGRAPH_VLOG(5) << "[NGTF-OPTIMIZER] Cannot construct an IdentityN node";
      }
    }
  }

  //
  // Variable capture: Part that replaces all instances of VariableV2 with the
  // NGraphVariable op. Making this replacement allows us to substitute in a
  // kernel that tracks the freshness of variables (invalidating freshness when
  // the reference is handed off to an "untrusted" op).
  //

  // Do variable capture then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(CaptureVariables(&graph, skip_these_nodes));
  if (DumpCapturedGraphs()) {
    DumpGraphs(graph, idx, "captured", "Graph With Variables Captured");
  }

  //
  // Encapsulation: Part that rewrites the graph for nGraph operation.
  //
  // The part has several phases, each executed in sequence:
  //
  //   1. Marking [ngraph_mark_for_clustering.cc]
  //   2. Cluster Assignment [ngraph_assign_clusters.cc]
  //   3. Cluster Deassignment [ngraph_deassign_clusters.cc]
  //   4. Cluster Encapsulation [ngraph_encapsulate_clusters.cc] - currently
  //      part of the ngraph_rewrite_pass.cc to be executed after POST_REWRITE
  //
  // Between phases, graph dumps (in both .dot and .pbtxt format) may be
  // requested by setting the following environment variables:
  //
  //   NGRAPH_TF_DUMP_UNMARKED_GRAPHS=1      dumps graphs before phase 1
  //   NGRAPH_TF_DUMP_MARKED_GRAPHS=1        dumps graphs after phase 1
  //   NGRAPH_TF_DUMP_CLUSTERED_GRAPHS=1     dumps graphs after phase 2
  //   NGRAPH_TF_DUMP_DECLUSTERED_GRAPHS=1   dumps graphs after phase 3
  //   NGRAPH_TF_DUMP_ENCAPSULATED_GRAPHS=1  dumps graphs after phase 4
  //   NGRAPH_TF_DUMP_GRAPHS=1               all of the above
  //

  // If requested, dump unmarked graphs.
  if (DumpUnmarkedGraphs()) {
    DumpGraphs(graph, idx, "unmarked", "Unmarked Graph");
  }

  // 1. Mark for clustering then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(MarkForClustering(&graph, skip_these_nodes));
  if (DumpMarkedGraphs()) {
    DumpGraphs(graph, idx, "marked", "Graph Marked for Clustering");
  }

  // 2. Assign clusters then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(AssignClusters(&graph));
  if (DumpClusteredGraphs()) {
    DumpGraphs(graph, idx, "clustered", "Graph with Clusters Assigned");
  }

  // 3. Deassign trivial clusters then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(DeassignClusters(&graph));
  if (DumpDeclusteredGraphs()) {
    DumpGraphs(graph, idx, "declustered",
               "Graph with Trivial Clusters De-Assigned");
  }

  // 4. Encapsulate clusters then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(EncapsulateClusters(&graph, idx));
  if (DumpEncapsulatedGraphs()) {
    DumpGraphs(graph, idx, "encapsulated", "Graph with Clusters Encapsulated");
  }

  // Rewrite for tracking then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(RewriteForTracking(&graph, idx));
  if (DumpTrackedGraphs()) {
    DumpGraphs(graph, idx, "tracked",
               "Graph with Variables Rewritten for Tracking");
  }

  // Convert the graph back to Graphdef
  graph.ToGraphDef(output);
  return Status::OK();
}

void NgraphOptimizer::Feedback(tensorflow::grappler::Cluster* cluster,
                               const tensorflow::grappler::GrapplerItem& item,
                               const GraphDef& optimize_output, double result) {
  // no-op
}

void NgraphOptimizer::DumpGraphs(Graph& graph, int idx,
                                 std::string filename_prefix,
                                 std::string title) {
  // If we have a "main" graph, dump that.
  auto dot_filename = DotFilename(filename_prefix, idx);
  auto pbtxt_filename = PbtxtFilename(filename_prefix, idx);
  NGRAPH_VLOG(0) << "[NGTF-OPTIMIZER] Dumping main graph to " << dot_filename;
  NGRAPH_VLOG(0) << "[NGTF-OPTIMIZER] Dumping main graph to " << pbtxt_filename;

  GraphToDotFile(&graph, dot_filename, title);
  GraphToPbTextFile(&graph, pbtxt_filename);
}

int NgraphOptimizer::FreshIndex() {
  mutex_lock l(s_serial_counter_mutex);
  return s_serial_counter++;
}

REGISTER_GRAPH_OPTIMIZER_AS(NgraphOptimizer, "ngraph-optimizer");

}  // end namespace ngraph_bridge

}  // end namespace tensorflow
