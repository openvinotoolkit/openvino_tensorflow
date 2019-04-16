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
#include "opexecuter.h"
#include <cstdlib>

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

// Utility Function to create NodeDef for _Arg and _Retval nodes
void OpExecuter::CreateNodeDef(const string op_type,
                               const string op_name_prefix, int index,
                               const DataType dt, NodeDef& node_def) {
  string new_node_name = op_name_prefix + std::to_string(index);
  node_def.set_name(new_node_name);
  node_def.set_op(op_type);
  SetAttrValue(dt, &((*(node_def.mutable_attr()))["T"]));
  SetAttrValue(index, &((*(node_def.mutable_attr()))["index"]));
}

// Update data structures for book keeping
// node_inedge_md : Map of
//                  key : Node*
//                  value : vector of pair{Node* src_incoming_edge, int
//                  src_output_index}
// node_outedge_md : Map of
//                  key : Node*
//                  value : vector of pair{Node* dst_outgoing_edge, int
//                  dst_input_index}
// node_outedges : Map of
//                  key : Node*
//                  value : vector of outgoing edges (Edge*)
// test_op : update pointer to test_op pointer
void OpExecuter::GetNodeData(Graph& graph, NodeMetaData& node_inedge_md,
                             NodeMetaData& node_outedge_md,
                             NodeOutEdges& node_outedges, Node** test_op) {
  bool found_test_op = false;
  for (const Edge* e : graph.edges()) {
    if (!found_test_op) {
      if (e->src()->IsOp() && (e->src()->type_string()) == test_op_type_) {
        found_test_op = true;
        *test_op = e->src();
      }
      if (e->dst()->IsOp() && (e->dst()->type_string()) == test_op_type_) {
        found_test_op = true;
        *test_op = e->dst();
      }
    }
    NGRAPH_VLOG(5) << "Edge between, Src: " << e->src()->name()
                   << " Src op index " << e->src_output()
                   << " ,Dst: " << e->dst()->name() << " dst ip index "
                   << e->dst_input();
    // update src's outedge metadata
    node_outedge_md[e->src()].push_back({e->dst(), e->dst_input()});
    node_inedge_md[e->dst()].push_back({e->src(), e->src_output()});
    node_outedges[e->src()].push_back(e);
  }
}

// Validate that the graph has N allowed_nodes and 1 test_op_type node
// Graph must look like this
//
// Const1     ConstN
//   \    ...    /
//    \         /
//      Test_Op
//
// TODO check for vector allowed_nodes
// when we allow other than "Const" node type as input
// Make allowed_nodes const member of the class, use set
void OpExecuter::ValidateGraph(const Graph& graph,
                               const vector<string> allowed_nodes) {
  NGRAPH_VLOG(5) << "Validate graph";
  bool found_test_op = false;
  Node* test_op = nullptr;
  for (Node* node : graph.nodes()) {
    if (node->IsSource() || node->IsSink()) {
      continue;
    } else if (node->type_string() == test_op_type_) {
      // only one node of type test_op
      ASSERT_FALSE(found_test_op) << "Only one op of type " << test_op_type_
                                  << " should exist in the graph. Found nodes "
                                  << node->name() << " and " << test_op->name();
      found_test_op = true;
      test_op = node;
    } else {
      ASSERT_TRUE(node->type_string() == allowed_nodes[0])
          << "Op of type " << node->type_string()
          << " not allowed in the graph. Found " << node->name();
    }
  }

  ASSERT_TRUE(found_test_op) << "Not found test_op : " << test_op_type_;

  NGRAPH_VLOG(5) << "Validate graph done";
}  // namespace testing

// Constructor Function
// TODO: Add support for ops that take static inputs
// currently static_input_map is empty
OpExecuter::OpExecuter(const Scope sc, const string test_op,
                       const vector<int>& static_input_indexes,
                       const vector<DataType>& op_types,
                       const vector<Output>& sess_run_fetchops)
    : tf_scope_(sc),
      test_op_type_(test_op),
      static_input_indexes_(static_input_indexes.begin(),
                            static_input_indexes.end()),
      expected_output_datatypes_(op_types),
      sess_run_fetchoutputs_(sess_run_fetchops) {}

// Destructor
OpExecuter::~OpExecuter() {}

void OpExecuter::RunTest(const string& ng_backend_name, float rtol,
                         float atol) {
  vector<Tensor> ngraph_outputs;
  ExecuteOnNGraph(ngraph_outputs, ng_backend_name);
  vector<Tensor> tf_outputs;
  ExecuteOnTF(tf_outputs);

  // Override the test result tolerance
  if (std::getenv("NGRAPH_TF_UTEST_RTOL") != nullptr) {
    rtol = std::atof(std::getenv("NGRAPH_TF_UTEST_RTOL"));
  }

  if (std::getenv("NGRAPH_TF_UTEST_ATOL") != nullptr) {
    atol = std::atof(std::getenv("NGRAPH_TF_UTEST_ATOL"));
  }

  Compare(tf_outputs, ngraph_outputs, rtol, atol);
}

// Uses tf_scope to execute on TF
void OpExecuter::ExecuteOnTF(vector<Tensor>& tf_outputs) {
  DeactivateNGraph();
  ClientSession session(tf_scope_);
  ASSERT_EQ(Status::OK(), session.Run(sess_run_fetchoutputs_, &tf_outputs))
      << "Failed to run opexecutor on TF";
  for (int i = 0; i < tf_outputs.size(); i++) {
    NGRAPH_VLOG(5) << " TF op " << i << tf_outputs[i].DebugString();
  }
}

// This function does the following:
// 1. Validates the graph
// 2. Rewrites the graph to have _Arg and _Retval nodes
//
// _Arg1    _ArgN
//   \   ...  /
//    \      /
//     Test_Op
//     /     \ 
//    /  ...  \ 
// _Retval1   _RetvalM
//
// 3. Gets Tensor values from Const Nodes for inputs to ng::Function call
// 4. Creates ng::Function
// 5. Executes ng::Function on CPU backend
// 6. Updates output of ng::Function into ngraph_output
// TODO : Refactor
void OpExecuter::ExecuteOnNGraph(vector<Tensor>& ngraph_outputs,
                                 const string& ng_backend_name) {
  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(tf_scope_.ToGraph(&graph));

  // For debug
  if (std::getenv("NGRAPH_TF_DUMP_GRAPHS") != nullptr) {
    GraphToPbTextFile(&graph, "unit_test_tf_graph_" + test_op_type_ + ".pbtxt");
  }

  ValidateGraph(graph, {"Const"});

  NodeMetaData node_inedge_metadata;
  NodeMetaData node_outedge_metadata;
  NodeOutEdges node_out_edges;
  Node* test_op;

  GetNodeData(graph, node_inedge_metadata, node_outedge_metadata,
              node_out_edges, &test_op);
  NGRAPH_VLOG(5) << "Got graph data. Found op " << test_op->type_string();

  // Get Tensor input shapes and values from the const nodes
  int number_of_inputs = test_op->num_inputs();
  // TODO : Validate static_input_indexes < number_of_inputs
  vector<TensorShape> input_shapes;
  vector<DataType> input_dt;
  vector<Tensor> tf_inputs;
  vector<const Tensor*> static_input_map;
  vector<Node*> input_node;

  for (int i = 0; i < number_of_inputs; i++) {
    Node* ip;
    ASSERT_EQ(Status::OK(), test_op->input_node(i, &ip))
        << "Failed to get input number " << i << " named " << ip->name();
    input_node.push_back(ip);

    Tensor ip_tensor;
    ASSERT_EQ(Status::OK(), GetNodeAttr(ip->attrs(), "value", &ip_tensor))
        << "Failed to get values from input number " << i << " named "
        << ip->name();
    input_shapes.push_back(ip_tensor.shape());
    input_dt.push_back(ip_tensor.dtype());
    tf_inputs.push_back(ip_tensor);

    NGRAPH_VLOG(5) << " Extracted tensor  " << i << " "
                   << ip_tensor.DebugString();
  }

  // Update static_input_map
  for (int i = 0; i < number_of_inputs; i++) {
    if (static_input_indexes_.find(i) != static_input_indexes_.end()) {
      static_input_map.push_back(&tf_inputs[i]);
      NGRAPH_VLOG(5) << "reading static tensor ptr " << i << " "
                     << (static_input_map[i])->DebugString();
    } else {
      static_input_map.push_back(nullptr);
    }
  }

  NGRAPH_VLOG(5) << "Got input nodes and tensors";

  // Replace the input nodes to Test_op with _Arg nodes
  for (int i = 0; i < number_of_inputs; i++) {
    Node* ip_node = input_node[i];
    NodeDef new_arg_node_def;
    CreateNodeDef("_Arg", "arg_", i, input_dt[i], new_arg_node_def);

    // Add node to graph
    Status status;
    Node* arg_node = graph.AddNode(new_arg_node_def, &status);
    ASSERT_EQ(Status::OK(), status) << "Failed to add node " << i << " named "
                                    << ip_node->name();

    // Remove the Const Node
    graph.RemoveNode(input_node[i]);

    // Add edge from SOURCE to _Arg
    auto src_nodes_metadata = node_inedge_metadata[ip_node];
    for (int j = 0; j < src_nodes_metadata.size(); j++) {
      graph.AddEdge(src_nodes_metadata[j].first, src_nodes_metadata[j].second,
                    arg_node, Graph::kControlSlot);
    }
    // Adds an edge from arg_node to test_op
    graph.AddEdge(arg_node, 0, test_op, i);
  }

  NGRAPH_VLOG(5) << "Replaced input nodes with _Arg";

  // Add _Retval to graph
  int number_of_outputs = expected_output_datatypes_.size();
  // For all the output edges from test_op (there should be only one, to SINK)
  // get the dest node and the
  // destination_input_index
  // (TODO : ) ADD ASSERT to check one?
  auto dest_nodes_metadata = node_outedge_metadata[test_op];

  // Remove edges from test_op to SINK (not removing might be also ok)
  for (const Edge* e : node_out_edges[test_op]) {
    graph.RemoveEdge(e);
  }

  for (int i = 0; i < number_of_outputs; i++) {
    // Add new retval_ node
    NodeDef new_ret_node_def;
    CreateNodeDef("_Retval", "retval_", i, expected_output_datatypes_[i],
                  new_ret_node_def);
    Status status;
    Node* ret_node = graph.AddNode(new_ret_node_def, &status);
    ASSERT_EQ(Status::OK(), status) << "Failed to add output " << i
                                    << " of type _Retval";

    // Add edges from _Retval to sink
    for (int j = 0; j < dest_nodes_metadata.size(); j++) {
      graph.AddEdge(ret_node, Graph::kControlSlot, dest_nodes_metadata[j].first,
                    dest_nodes_metadata[j].second);
    }
    // Add edges from test_op to _Retval
    graph.AddEdge(test_op, i, ret_node, 0);
  }

  NGRAPH_VLOG(5) << "Added _Retval nodes ";

  NGRAPH_VLOG(5) << "After rewrite *** ";
  for (const Edge* e : graph.edges()) {
    NGRAPH_VLOG(5) << "Edge between, Src: " << e->src()->name()
                   << " ,Dst: " << e->dst()->name();
  }
  // For debug
  if (std::getenv("NGRAPH_TF_DUMP_GRAPHS") != nullptr) {
    GraphToPbTextFile(&graph,
                      "unit_test_rewrite_ngraph_" + test_op_type_ + ".pbtxt");
  }

  // Create nGraph function
  NGRAPH_VLOG(5) << " Create ng function ";
  ASSERT_EQ(Status::OK(),
            Builder::TranslateGraph(input_shapes, static_input_map, &graph,
                                    ng_function))
      << "Failed to TranslateGraph";

  // ng function should get same number of outputs
  ASSERT_EQ(expected_output_datatypes_.size(), ng_function->get_output_size())
      << "Number of outputs of requested outputs and ngraph function outputs "
         "do not match";

  // For debug
  // Serialize to nGraph if needed
  if (std::getenv("NGRAPH_ENABLE_SERIALIZE") != nullptr) {
    NgraphSerialize("unit_test_" + test_op_type_ + ".json", ng_function);
  }

  // Create nGraph backend
  // If NGRAPH_TF_BACKEND is set create that backend
  // Else create backend of type ng_backend_name
  string ng_backend_type = ng_backend_name;
  const char* ng_backend_env_value = std::getenv("NGRAPH_TF_BACKEND");
  if (ng_backend_env_value != nullptr) {
    string backend_env = std::string(ng_backend_env_value);
    bool valid_ngraph_tf_backend =
        !backend_env.empty() && BackendManager::IsSupportedBackend(backend_env);
    ASSERT_TRUE(valid_ngraph_tf_backend) << "NGRAPH_TF_BACKEND " << backend_env
                                         << " is not a supported backend";
    ng_backend_type = backend_env;
  }

  NGRAPH_VLOG(5) << " Creating NG Backend " << ng_backend_type;
  BackendManager::CreateBackend(ng_backend_type);
  auto backend = BackendManager::GetBackend(ng_backend_type);

  // Allocate tensors for inputs
  vector<std::shared_ptr<ngraph::runtime::Tensor>> ng_ip_tensors;
  vector<std::shared_ptr<ngraph::runtime::Tensor>> ng_op_tensors;

  NGRAPH_VLOG(5) << " Creating ng inputs ";
  NGRAPH_VLOG(5) << "No of inputs " << tf_inputs.size();
  for (int i = 0; i < tf_inputs.size(); i++) {
    ng::Shape ng_shape;
    ASSERT_EQ(Status::OK(),
              TFTensorShapeToNGraphShape(tf_inputs[i].shape(), &ng_shape))
        << "Shape of " << i << "th input did not match";
    ng::element::Type ng_et;
    ASSERT_EQ(Status::OK(),
              TFDataTypeToNGraphElementType(tf_inputs[i].dtype(), &ng_et))
        << "Datatype of " << i << "th input is "
        << DataTypeString(tf_inputs[i].dtype()) << ". Ngraph's element type is "
        << ng_et;

    void* src_ptr = (void*)DMAHelper::base(&tf_inputs[i]);
    std::shared_ptr<ngraph::runtime::Tensor> result;
    if (ng_backend_type != "CPU") {
      result = backend->create_tensor(ng_et, ng_shape);
      result->write(src_ptr, 0, result->get_element_count() * ng_et.size());
    } else {
      result = backend->create_tensor(ng_et, ng_shape, src_ptr);
    }

    ng_ip_tensors.push_back(result);
  }

  NGRAPH_VLOG(5) << " Creating ng outputs ";
  vector<TensorShape> tf_op_shapes;
  for (int i = 0; i < number_of_outputs; i++) {
    auto ng_op_shape = ng_function->get_output_shape(i);
    auto ng_op_type = ng_function->get_output_element_type(i);

    ng::element::Type ng_et_expected;
    ASSERT_EQ(Status::OK(), TFDataTypeToNGraphElementType(
                                expected_output_datatypes_[i], &ng_et_expected))
        << "Datatype of " << i << "th output is "
        << DataTypeString(expected_output_datatypes_[i])
        << ". Ngraph's element type is " << ng_et_expected;

    // Expected element type should match ng_op_type
    ASSERT_EQ(ng_et_expected, ng_op_type)
        << "Expected Ngraph datatype of " << i << "th output is "
        << ng_et_expected << ", but ngraph function has " << ng_op_type;
    vector<int64> dims;
    for (auto dim : ng_op_shape) {
      dims.push_back(dim);
    }
    TensorShape tf_shape(dims);
    tf_op_shapes.push_back(tf_shape);
    auto result = backend->create_tensor(ng_op_type, ng_op_shape);
    ng_op_tensors.push_back(result);
  }

  // Execute the nGraph
  NGRAPH_VLOG(5) << " Executing on nGraph ";
  BackendManager::LockBackend(ng_backend_type);
  try {
    auto exec = backend->compile(ng_function);
    exec->call(ng_op_tensors, ng_ip_tensors);
  } catch (const std::exception& exp) {
    BackendManager::UnlockBackend(ng_backend_type);
    NgraphSerialize("unit_test_error_" + test_op_type_ + ".json", ng_function);
    FAIL() << "Exception while executing on nGraph " << exp.what();
  } catch (...) {
    BackendManager::UnlockBackend(ng_backend_type);
    NgraphSerialize("unit_test_error_" + test_op_type_ + ".json", ng_function);
    FAIL() << "Exception while executing on nGraph";
  }
  BackendManager::UnlockBackend(ng_backend_type);

  NGRAPH_VLOG(5) << " Writing to Tensors ";
  for (auto i = 0; i < ng_function->get_output_size(); i++) {
    // Convert to tf tensor
    Tensor output_tensor(expected_output_datatypes_[i], tf_op_shapes[i]);
    void* dst_ptr = DMAHelper::base(&output_tensor);
    ng_op_tensors[i]->read(dst_ptr, 0, output_tensor.TotalBytes());
    ngraph_outputs.push_back(output_tensor);
    NGRAPH_VLOG(5) << " NGRAPH op " << i << ngraph_outputs[i].DebugString();
  }

  // Clear the tesnors
  ng_ip_tensors.clear();
  ng_op_tensors.clear();

  // Release the backend
  BackendManager::ReleaseBackend(ng_backend_type);

}  // ExecuteOnNGraph

}  // namespace testing
}  // namespace ngraph_bridge

}  // namespace tensorflow
