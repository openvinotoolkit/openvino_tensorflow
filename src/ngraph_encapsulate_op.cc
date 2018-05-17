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

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"

#include "ngraph_cluster_manager.h"

namespace tf = tensorflow;


namespace ngraph_bridge {
extern const char* const DEVICE_NGRAPH_CPU;

REGISTER_OP("NGraphEncapsulate")
    .Input("args: Targuments")
    .Attr("Targuments: list(type) >= 0")
    .Output("results: Tresults")
    .Attr("Tresults: list(type) >= 0")
    .Attr("ngraph_cluster: int")
    .SetIsStateful()
    .Doc("nGraph Encapsulation Op. For use by the nGraph JIT only.");

class NGraphEncapsulateOp : public tf::OpKernel {
 public:
  explicit NGraphEncapsulateOp(tf::OpKernelConstruction* ctx)
      : tf::OpKernel(ctx)
      , m_graph(tf::OpRegistry::Global()) {
    int ngraph_cluster;
    tf::GraphDef* graph_def;

    // TODO(amprocte): need to check status result here.
    tf::Status status = ctx->GetAttr<int>("ngraph_cluster",&ngraph_cluster);
    graph_def = NGraphClusterManager::GetClusterGraph(ngraph_cluster);

    tf::GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    // TODO(amprocte): need to check status result here.
    tf::ConvertGraphDefToGraph(opts, *graph_def, &m_graph);
    // DataTypeVector constant_types;
    // OP_REQUIRES_OK(ctx, ctx->GetAttr("Tconstants", &constant_types));
    // num_constant_args_ = constant_types.size();
    // OP_REQUIRES_OK(ctx, ctx->GetAttr("Nresources", &num_resource_args_));
    VLOG(0) << "NGraphEncapsulateOp::Number of inputs: " << ctx->num_inputs();
    VLOG(0) << "NGraphEncapsulateOp::Number of outputs: " << ctx->num_outputs();
  }
  ~NGraphEncapsulateOp() override {
    // d-tor
  }
  void Compute(tf::OpKernelContext* ctx) override {
  }
 private:
  tf::Graph m_graph;
};

}  // namespace ngraph_bridge

namespace tensorflow {
REGISTER_KERNEL_BUILDER(Name("NGraphEncapsulate").Device(ngraph_bridge::DEVICE_NGRAPH_CPU),
                        ngraph_bridge::NGraphEncapsulateOp);
}
