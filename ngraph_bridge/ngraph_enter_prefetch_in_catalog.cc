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

#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/serializer.hpp"

#include "logging/ngraph_log.h"
#include "ngraph_bridge/ngraph_catalog.h"
#include "ngraph_bridge/ngraph_enter_prefetch_in_catalog.h"
#include "ngraph_bridge/ngraph_prefetch_shared_data.h"
#include "ngraph_bridge/ngraph_utils.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

// Populate the PrefetchedInputIndexMap

// We collect the below information for the catalog
// 1. If the input to "NGraphEncapsulate" node
// is coming form IteratorGetNext, catalog it
// i.e. we add the node_name and the input indexes for the
// "NGraphEncapsulate" node to the PrefetchedInputIndexMap
// We add mapping of {graphId_nodename : (input_indexs)} to the
// PrefetchedInputIndexMap
//

Status EnterPrefetchInCatalog(Graph* graph, int graph_id) {
  if (std::getenv(NGraphPrefetchSharedResouce::NGRAPH_TF_USE_PREFETCH) ==
      nullptr) {
    // if prefetch is not requested return
    return Status::OK();
  }

  // Go over all the nodes in the graph
  for (auto node : graph->op_nodes()) {
    // If the node is a NGraphEncapsulate, go over all it's
    // inputs
    map<int, int> in_indexes_for_encap;
    if (node->type_string() == "NGraphEncapsulate") {
      for (auto edge : node->in_edges()) {
        // If any input is coming from "IteratorGetNext" then
        // add the input index for it to the set
        // [TODO] Data Pipeling assumes there is only 1 IteratorGetNext
        //
        if (edge->src()->type_string() == "IteratorGetNext") {
          NGRAPH_VLOG(4) << "Adding to PrefetchedInputIndexMap";
          NGRAPH_VLOG(4) << "Key: " << node->name();
          NGRAPH_VLOG(4) << "NGEncap Input index: " << edge->dst_input();
          NGRAPH_VLOG(4) << "IteratorGetNext Output index: "
                         << edge->src_output();
          in_indexes_for_encap.insert({edge->dst_input(), edge->src_output()});
        }
      }  // end loop over input edges

      if (in_indexes_for_encap.size() > 0) {
        try {
          NGraphCatalog::AddToPrefetchedInputIndexMap(graph_id, node->name(),
                                                      in_indexes_for_encap);
        } catch (const std::exception& exp) {
          return errors::Internal(
              "Caught exception while entering in catalog: ", exp.what(), "\n");
        }
      }
    }
  }  // end loop over graph nodes
  NGRAPH_VLOG(4) << "Entered in Catalog";
  return Status::OK();
}  // enter in catalog

}  // namespace ngraph_bridge

}  // namespace tensorflow
