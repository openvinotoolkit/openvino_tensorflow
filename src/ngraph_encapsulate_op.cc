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

#include "ngraph_builder.h"
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
      : tf::OpKernel(ctx), m_graph(tf::OpRegistry::Global()) {
    int ngraph_cluster;
    tf::GraphDef* graph_def;

    // TODO(amprocte): need to check status result here.
    OP_REQUIRES_OK(ctx, ctx->GetAttr<int>("ngraph_cluster", &ngraph_cluster));
    graph_def = NGraphClusterManager::GetClusterGraph(ngraph_cluster);

    tf::GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    // TODO(amprocte): need to check status result here.
    OP_REQUIRES_OK(ctx, tf::ConvertGraphDefToGraph(opts, *graph_def, &m_graph));

    VLOG(0) << "NGraphEncapsulateOp::Number of inputs: " << ctx->num_inputs();
    VLOG(0) << "NGraphEncapsulateOp::Number of outputs: " << ctx->num_outputs();
  }

  ~NGraphEncapsulateOp() override {
    // d-tor
  }

  void Compute(tf::OpKernelContext* ctx) override {
    VLOG(0) << "NGraphMulOp::Compute() Step: " << ctx->step_id()
            << " Op: " << ctx->op_kernel().name();
    VLOG(0) << "Inputs: " << ctx->num_inputs()
            << " Outputs: " << ctx->num_outputs();
    // Get the inputs
    std::vector<tf::TensorShape> input_shapes;
    for (int i = 0; i < ctx->num_inputs(); i++) {
      const tf::Tensor& input_tensor = ctx->input(i);
      input_shapes.push_back(input_tensor.shape());
    }

    // Compile the graph using nGraph
    auto ng_function =
        ngraph_bridge::Builder::TranslateGraph(input_shapes, &m_graph);
    OP_REQUIRES(
        ctx, ng_function != nullptr,
        tf::errors::InvalidArgument("Cannot convert TF graph to nGraph"));

    // Create the nGraph backend
    auto backend = ng::runtime::Backend::create("CPU");

    // Allocate tensors for arguments a, b, c
    vector<shared_ptr<ng::runtime::TensorView>> ng_inputs;
    for (int i = 0; i < input_shapes.size(); i++) {
      ng::Shape ng_shape(input_shapes[i].dims());
      for (int j = 0; j < input_shapes[i].dims(); ++j) {
        ng_shape[j] = input_shapes[i].dim_size(j);
      }

      auto t_x = backend->create_tensor(ng::element::f32, ng_shape);
      float v_x[2][3] = {{1, 1, 1}, {1, 1, 1}};
      t_x->write(&v_x, 0, sizeof(v_x));
      ng_inputs.push_back(t_x);
    }

    // Allocate tensor for the result(s)
    vector<shared_ptr<ng::runtime::TensorView>> outputs;
    for (auto i = 0; i < ng_function->get_output_size(); i++) {
      auto shape = ng_function->get_output_shape(i);
      auto elem_type = ng_function->get_output_element_type(i);
      auto t_result = backend->create_tensor(elem_type, shape);
      outputs.push_back(t_result);
    }

    // Execute the nGraph function.
    cout << "Calling nGraph function\n";
    backend->call(ng_function, outputs, ng_inputs);

    // Save the output
    // First determine the outpuit shapes
    // Allocate tensor for the result(s)
    // vector<shared_ptr<ng::runtime::TensorView>> outputs;
    for (auto i = 0; i < ng_function->get_output_size(); i++) {
      auto shape = ng_function->get_output_shape(i);
      vector<long long int> dims;
      for (auto dim : shape) {
        dims.push_back(dim);
      }
      tf::TensorShape tf_shape(dims);
      // Create the TF output tensors
      tf::Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, tf_shape, &output_tensor));

      // auto t_result = backend->create_tensor(elem_type, shape);
      // outputs.push_back(t_result);

      auto elem_type = ng_function->get_output_element_type(i);

      // Create the TF output tensors
      // tf::Tensor* output_tensor = nullptr;
      // OP_REQUIRES_OK(
      //     ctx, ctx->allocate_output(0, input_shapes.shape(),
      //     &output_tensor));

      // auto t_result = backend->create_tensor(elem_type, shape);
      // outputs.push_back(t_result);
    }
  }

 private:
  tf::Graph m_graph;
};

}  // namespace ngraph_bridge

namespace tensorflow {
REGISTER_KERNEL_BUILDER(
    Name("NGraphEncapsulate").Device(ngraph_bridge::DEVICE_NGRAPH_CPU),
    ngraph_bridge::NGraphEncapsulateOp);
}
