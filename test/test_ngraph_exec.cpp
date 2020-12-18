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
#include <regex>
#include "gtest/gtest.h"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/platform/env.h"

#include "ngraph_bridge/backend_manager.h"
#include "ngraph_bridge/ngraph_builder.h"
#include "ngraph_bridge/ngraph_utils.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset.hpp"
#include "ngraph_bridge/default_opset.h"

#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

static int FindNumberOfNodes(const Graph* graph, const string op_type) {
  int count = 0;
  for (auto node : graph->nodes()) {
    if (node->type_string() == op_type) {
      count++;
    }
  }
  return count;
}

class NGraphExecTest : public ::testing::Test {
 protected:
  // Loads the .pbtxt into a graph object
  Status LoadGraph(const string& graph_pbtxt_file, Graph* graph) {
    GraphDef gdef;
    TF_RETURN_IF_ERROR(ReadTextProto(Env::Default(), graph_pbtxt_file, &gdef));
    GraphConstructorOptions opts;

    // Set the allow_internal_ops to true so that graphs with node names such as
    // _arg_Placeholder_1_0_1_0_arg are allowed. These op names are generated
    // during the graph rewrite passes and considered internal
    opts.allow_internal_ops = true;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, gdef, graph));
    return Status::OK();
  }

  // Translates the TFGraph into NGFunction assumes no static inputs
  Status TranslateTFGraphNoStatic(const vector<TensorShape>& tf_input_shapes,
                                  const Graph& input_graph,
                                  shared_ptr<ngraph::Function>& ng_function) {
    // Translate the Graph: Create ng_function
    std::vector<const Tensor*> static_input_map(tf_input_shapes.size(),
                                                nullptr);
    TF_RETURN_IF_ERROR(ngraph_bridge::Builder::TranslateGraph(
        tf_input_shapes, static_input_map, &input_graph, "test_ngraph_exec",
        ng_function));
    return Status::OK();
  }

  // Executes the graph on TF
  Status RunGraphOnTF(const Graph& graph,
                      const vector<pair<string, Tensor>>& feed_dict,
                      const vector<string>& out_node_names,
                      vector<Tensor>& out_tensors) {
    // Create Session
    SessionOptions options;
    options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_opt_level(tf::OptimizerOptions_Level_L0);
    options.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_constant_folding(tf::RewriterConfig::OFF);
    std::unique_ptr<Session> session(NewSession(options));

    // Attach Graph
    GraphDef gdef;
    graph.ToGraphDef(&gdef);
    TF_RETURN_IF_ERROR(session->Create(gdef));
    DeactivateNGraph();
    Status status = session->Run(feed_dict, out_node_names, {}, &out_tensors);
    ActivateNGraph();
    return status;
  }

  shared_ptr<Graph> attach_retval_node(Scope& scope, Graph* pgraph, Node* op,
                                       int outidx = 0) {
    Status s;
    GraphDefBuilder::Options gdopts(pgraph, &s);
    // ASSERT_OK(s);
    if (!s.ok()) LOG(FATAL) << s.error_message();
    string name = op->name() + "_retval" + to_string(outidx);
    auto dtype = op->output_type(outidx);
    NodeBuilder nbdlr(gdopts.WithName(name)
                          .WithAttr("T", dtype)
                          .WithAttr("index", outidx)
                          .GetNameForOp(""),
                      "_Retval", gdopts.op_registry());
    nbdlr.Input(op, outidx).Attr("index", outidx);
    auto ret = gdopts.FinalizeBuilder(&nbdlr);
    if (ret == nullptr) LOG(FATAL) << "FinalizeBuilder failed!";
    GraphDef gdef;
    s = scope.ToGraphDef(&gdef);
    if (!s.ok()) LOG(FATAL) << s.error_message();
    GraphConstructorOptions gcopts;
    gcopts.allow_internal_ops = true;
    auto pgraph_new = make_shared<Graph>(pgraph->op_registry());
    ConvertGraphDefToGraph(gcopts, gdef, pgraph_new.get());
    return pgraph_new;
  }

  void expect_const_count_ngfunc(const Graph& g, int expected) {
    std::vector<TensorShape> tf_input_shapes;
    shared_ptr<ng::Function> func;
    ASSERT_OK(TranslateTFGraphNoStatic(tf_input_shapes, g, func));
    int numconst = 0;
    for (const auto& node : func->get_ops()) {
      if (ngraph::is_type<opset::Constant>(node)) numconst++;
    }
    ASSERT_EQ(numconst, expected);
  };
};

TEST_F(NGraphExecTest, Axpy) {
  auto env_map = StoreEnv({"NGRAPH_TF_BACKEND"});
  SetBackendUsingEnvVar("CPU");

  Graph input_graph(OpRegistry::Global());
  ASSERT_OK(LoadGraph("test_axpy_launchop.pbtxt", &input_graph));

  // Create the inputs for this graph
  Tensor x(DT_FLOAT, TensorShape({2, 3}));
  Tensor y(DT_FLOAT, TensorShape({2, 3}));

  std::vector<TensorShape> input_shapes;
  input_shapes.push_back(x.shape());
  input_shapes.push_back(y.shape());

  shared_ptr<ng::Function> ng_function;
  ASSERT_OK(TranslateTFGraphNoStatic(input_shapes, input_graph, ng_function));

  // Create the nGraph backend
  auto backend = BackendManager::GetBackend();
  ASSERT_NE(backend, nullptr);

  // Allocate tensors for arguments a, b, c
  ng::Shape ng_shape_x(x.shape().dims());
  for (int i = 0; i < x.shape().dims(); ++i) {
    ng_shape_x[i] = x.shape().dim_size(i);
  }

  ng::Shape ng_shape_y(y.shape().dims());
  for (int i = 0; i < y.shape().dims(); ++i) {
    ng_shape_y[i] = y.shape().dim_size(i);
  }

  auto t_x = make_shared<IETensor>(ng::element::f32, ng_shape_x);
  float v_x[2][3] = {{1, 1, 1}, {1, 1, 1}};
  t_x->write(&v_x, sizeof(v_x));

  auto t_y = make_shared<IETensor>(ng::element::f32, ng_shape_y);
  t_y->write(&v_x, sizeof(v_x));

  // Execute the nGraph function.
  auto exec = backend->Compile(ng_function);
  vector<shared_ptr<ng::runtime::Tensor>> outputs;
  exec->Call({t_x, t_y}, outputs);

  for (size_t i = 0; i < ng_function->get_output_size(); i++) {
    DumpNGTensor<float>(cout, ng_function->get_output_op(i)->get_name(),
                        outputs[i]);
    cout << endl;
  }
  // Add the validation logic
  // TODO
  RestoreEnv(env_map);
}

TEST_F(NGraphExecTest, Axpy8bit) {
  auto env_map = StoreEnv({"NGRAPH_TF_BACKEND"});
  SetBackendUsingEnvVar("CPU");

  Graph input_graph(OpRegistry::Global());
  ASSERT_OK(LoadGraph("test_axpy_int8_launchop.pbtxt", &input_graph));

  // Create the inputs for this graph
  Tensor x(DT_INT8, TensorShape({2, 2}));
  Tensor y(DT_INT8, TensorShape({2, 2}));

  std::vector<TensorShape> input_shapes;
  input_shapes.push_back(x.shape());
  input_shapes.push_back(y.shape());

  shared_ptr<ng::Function> ng_function;
  ASSERT_OK(TranslateTFGraphNoStatic(input_shapes, input_graph, ng_function));

  // Create the nGraph backend
  auto backend = BackendManager::GetBackend();
  ASSERT_NE(backend, nullptr);

  // Allocate tensors for arguments a, b, c
  ng::Shape ng_shape_x(x.shape().dims());
  for (int i = 0; i < x.shape().dims(); ++i) {
    ng_shape_x[i] = x.shape().dim_size(i);
  }

  ng::Shape ng_shape_y(y.shape().dims());
  for (int i = 0; i < y.shape().dims(); ++i) {
    ng_shape_y[i] = y.shape().dim_size(i);
  }

  auto t_x = make_shared<IETensor>(ng::element::i8, ng_shape_x);
  int8 v_x[2][2] = {{1, 1}, {1, 1}};
  t_x->write(&v_x, sizeof(v_x));

  auto t_y = make_shared<IETensor>(ng::element::i8, ng_shape_y);
  t_y->write(&v_x, sizeof(v_x));

  // Execute the nGraph function.
  auto exec = backend->Compile(ng_function);
  vector<shared_ptr<ng::runtime::Tensor>> outputs;
  exec->Call({t_x, t_y}, outputs);

  for (size_t i = 0; i < ng_function->get_output_size(); i++) {
    DumpNGTensor<int8>(cout, ng_function->get_output_op(i)->get_name(),
                       outputs[i]);
    cout << endl;
  }
  // Add the validation logic
  // TODO
  RestoreEnv(env_map);
}

TEST_F(NGraphExecTest, FindNumberOfNodesUtil1) {
  Graph input_graph(OpRegistry::Global());
  ASSERT_OK(LoadGraph("test_axpy_launchop.pbtxt", &input_graph));

  int number_of_args = FindNumberOfNodes(&input_graph, "_Arg");
  int number_of_retvals = FindNumberOfNodes(&input_graph, "_Retval");
  int number_of_const = FindNumberOfNodes(&input_graph, "Const");
  int number_of_xyz = FindNumberOfNodes(&input_graph, "XYZ");

  ASSERT_EQ(number_of_args, 2);
  ASSERT_EQ(number_of_retvals, 2);
  ASSERT_EQ(number_of_const, 1);
  ASSERT_EQ(number_of_xyz, 0);
}

TEST_F(NGraphExecTest, FindNumberOfNodesUtil2) {
  Graph input_graph(OpRegistry::Global());
  ASSERT_OK(LoadGraph("test_general_graph.pbtxt", &input_graph));

  int number_of_args = FindNumberOfNodes(&input_graph, "_Arg");
  int number_of_retvals = FindNumberOfNodes(&input_graph, "_Retval");
  int number_of_add = FindNumberOfNodes(&input_graph, "Add");
  int number_of_sub = FindNumberOfNodes(&input_graph, "Sub");

  ASSERT_EQ(number_of_args, 3);
  ASSERT_EQ(number_of_retvals, 3);
  ASSERT_EQ(number_of_add, 2);
  ASSERT_EQ(number_of_sub, 1);
}

TEST_F(NGraphExecTest, NGraphPassConstantFolding1) {
  Graph input_graph(OpRegistry::Global());
  ASSERT_OK(LoadGraph("test_graph1.pbtxt", &input_graph));

  setenv("NGRAPH_TF_CONSTANT_FOLDING", "1", true);
  expect_const_count_ngfunc(input_graph, 1);
  unsetenv("NGRAPH_TF_CONSTANT_FOLDING");

  setenv("NGRAPH_TF_CONSTANT_FOLDING", "0", true);
  expect_const_count_ngfunc(input_graph, 3);
  unsetenv("NGRAPH_TF_CONSTANT_FOLDING");
}

TEST_F(NGraphExecTest, NGraphPassConstantFolding2) {
  Scope root = Scope::NewRootScope();
  Graph* pgraph = root.graph();
  Status s;
  auto a = ops::Const(root, {{1, 2}, {2, 4}});
  auto b = ops::Const(root, {{2, 2}, {1, 1}});
  auto add1 = ops::Add(root, a, b);
  auto c = ops::Const(root, {{0, 3}, {1, 1}});
  auto add2 = ops::Add(root, add1, c);
  // attach _Retval node
  auto pgraph_new = attach_retval_node(root, pgraph, add2.node());

  setenv("NGRAPH_TF_CONSTANT_FOLDING", "1", true);
  expect_const_count_ngfunc(*pgraph_new, 1);
  unsetenv("NGRAPH_TF_CONSTANT_FOLDING");

  setenv("NGRAPH_TF_CONSTANT_FOLDING", "0", true);
  expect_const_count_ngfunc(*pgraph_new, 3);
  unsetenv("NGRAPH_TF_CONSTANT_FOLDING");
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
