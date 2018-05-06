/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
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
#include "gtest/gtest.h"

#include "ngraph_builder.h"
#include "ngraph_utils.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"

using namespace std;
namespace tf = tensorflow;

TEST(graph_exec, builder) {
  tf::GraphDef gdef;
  auto status = tf::ReadTextProto(tf::Env::Default(), "test_py.pbtxt", &gdef);
  ASSERT_TRUE(status == tf::Status::OK()) << "Can't read protobuf graph";

  tf::Graph input_graph(tf::OpRegistry::Global());
  tf::GraphConstructorOptions opts;
  ASSERT_EQ(tf::ConvertGraphDefToGraph(opts, gdef, &input_graph),
            tf::Status::OK());
  auto ng_function = ngraph_bridge::Builder::TransformGraph(&input_graph);
}

TEST(graph_exec, DISABLED_simple) {
  tf::GraphDef gdef;
  auto status = tf::ReadTextProto(tf::Env::Default(), "test_py.pbtxt", &gdef);
  EXPECT_TRUE(status == tf::Status::OK()) << "Can't read protobuf graph";

  tf::Graph input_graph(tf::OpRegistry::Global());
  tf::GraphConstructorOptions opts;
  ASSERT_EQ(tf::ConvertGraphDefToGraph(opts, gdef, &input_graph),
            tf::Status::OK());

  for (const tf::Node* n : input_graph.nodes()) {
    cout << "Node: " << n->name() << " Type: " << n->type_string() << endl;
    for (const tf::Edge* edge : n->in_edges()) {
      cout << "\tEdge " << edge->src()->name() << " --> " << edge->dst()->name()
           << endl;
    }
  }
  // Do a topological sort
  vector<tf::Node*> ordered;
  tf::GetReversePostOrder(input_graph,
                          &ordered);  // This will give us topological sort.

  cout << "After Topo sort\n";
  for (auto n : ordered) {
    cout << "Node: " << n->name() << " Type: " << n->type_string() << endl;
    for (const tf::Edge* edge : n->in_edges()) {
      cout << "\tEdge " << edge->src()->name() << " --> " << edge->dst()->name()
           << endl;
    }
  }

  // Next create the nGraph graph

  GraphToPbTextFile("./test_graph.pbtxt", &input_graph);

  // graph.ToGraphDef(&g_def);

  // Write

  // DO some transformation

  // Verify
}