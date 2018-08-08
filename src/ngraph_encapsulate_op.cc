/*******************************************************************************
o * Copyright 2017-2018 Intel Corporation
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
#include <fstream>

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"

#include "ngraph/serializer.hpp"

#include "ngraph_builder.h"
#include "ngraph_cluster_manager.h"
#include "ngraph_log.h"
#include "ngraph_utils.h"

#include "ngraph/runtime/interpreter/int_backend.hpp"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

REGISTER_OP("NGraphEncapsulate")
    .Input("args: Targuments")
    .Attr("Targuments: list(type) >= 0")
    .Output("results: Tresults")
    .Attr("Tresults: list(type) >= 0")
    .Attr("ngraph_cluster: int")
    .SetIsStateful()
    .Doc("nGraph Encapsulation Op. For use by the nGraph JIT only.");

class NGraphEncapsulateOp : public OpKernel {
 public:
  explicit NGraphEncapsulateOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), m_graph(OpRegistry::Global()) {
    GraphDef* graph_def;

    // TODO(amprocte): need to check status result here.
    OP_REQUIRES_OK(ctx, ctx->GetAttr<int>("ngraph_cluster", &m_ngraph_cluster));
    graph_def = NGraphClusterManager::GetClusterGraph(m_ngraph_cluster);

    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    // TODO(amprocte): need to check status result here.
    OP_REQUIRES_OK(ctx, ConvertGraphDefToGraph(opts, *graph_def, &m_graph));

    // Create the backend
    if (m_ng_backend == nullptr) {
#if defined(NGRAPH_EMBEDDED_IN_TENSORFLOW)
      m_ng_backend = std::make_shared<ng::runtime::interpreter::INTBackend>();
#else
      m_ng_backend = ng::runtime::Backend::create("CPU");
#endif
      OP_REQUIRES(ctx, m_ng_backend != nullptr,
                  errors::InvalidArgument("Cannot create nGraph backend"));
    }
  }

  // TODO(amprocte): this needs to be made thread-safe (compilation cache OK?).
  void Compute(OpKernelContext* ctx) override {
    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute starting for cluster " << m_ngraph_cluster;

    // Get the inputs
    std::vector<TensorShape> input_shapes;
    std::stringstream signature_ss;
    for (int i = 0; i < ctx->num_inputs(); i++) {
      const Tensor& input_tensor = ctx->input(i);
      input_shapes.push_back(input_tensor.shape());
      for (const auto& x : input_tensor.shape()) {
        signature_ss << x.size << ",";
      }
      signature_ss << ";";
    }

    std::shared_ptr<ngraph::Function> ng_function;
    std::string signature = signature_ss.str();
    auto it = m_ng_functions.find(signature);

    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute got inputs for cluster " << m_ngraph_cluster;

    // Compile the graph using nGraph.
    //
    // TODO(amprocte): Investigate performance of the compilation cache.
    if (it == m_ng_functions.end()) {
      NGRAPH_VLOG(1) << "Compilation cache miss: " << ctx->op_kernel().name();
      OP_REQUIRES_OK(
          ctx, Builder::TranslateGraph(input_shapes, &m_graph, ng_function));

      // Serialize to nGraph if needed
      if (std::getenv("NGRAPH_ENABLE_SERIALIZE") != nullptr) {
        std::string file_name =
            "tf_function_" + ctx->op_kernel().name() + ".js";
        NGRAPH_VLOG(0) << "Serializing graph to: " << file_name;
        std::string js = ngraph::serialize(ng_function, 4);
        {
          std::ofstream f(file_name);
          f << js;
        }
      }

      m_ng_functions[signature] = ng_function;
    } else {
      ng_function = it->second;
    }

    // Allocate tensors for arguments.
    vector<shared_ptr<ng::runtime::TensorView>> ng_inputs;

    for (int i = 0; i < input_shapes.size(); i++) {
      ng::Shape ng_shape(input_shapes[i].dims());
      for (int j = 0; j < input_shapes[i].dims(); ++j) {
        ng_shape[j] = input_shapes[i].dim_size(j);
      }

      ng::element::Type ng_element_type;
      OP_REQUIRES_OK(ctx, TFDataTypeToNGraphElementType(ctx->input(i).dtype(),
                                                        &ng_element_type));

      void* src_ptr = (void*)DMAHelper::base(&ctx->input(i));
      auto t = m_ng_backend->create_tensor(ng_element_type, ng_shape, src_ptr);

      ng_inputs.push_back(t);
    }

    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute allocated argument tensors for cluster " << m_ngraph_cluster;

    // Allocate tensors for the results.
    vector<shared_ptr<ng::runtime::TensorView>> outputs;
    for (auto i = 0; i < ng_function->get_output_size(); i++) {
      auto shape = ng_function->get_output_shape(i);
      auto elem_type = ng_function->get_output_element_type(i);

      // Create the TF output tensor
      vector<int64> dims;
      for (auto dim : shape) {
        dims.push_back(dim);
      }
      TensorShape tf_shape(dims);
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, tf_shape, &output_tensor));

      // Make sure the nGraph-inferred element type agrees with what TensorFlow
      // expected.
      ng::element::Type expected_elem_type;
      OP_REQUIRES_OK(
          ctx, TFDataTypeToNGraphElementType(ctx->expected_output_dtype(i),
                                             &expected_elem_type));
      OP_REQUIRES(
          ctx, elem_type == expected_elem_type,
          errors::Internal("Element type inferred by nGraph does not match "
                           "the element type expected by TensorFlow"));

      // Create the nGraph output tensor
      void* dst_ptr = DMAHelper::base(output_tensor);
      auto t_result = m_ng_backend->create_tensor(elem_type, shape, dst_ptr);

      outputs.push_back(t_result);
    }

    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute allocated result tensors for cluster " << m_ngraph_cluster;

    // Execute the nGraph function.
    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute call starting for cluster " << m_ngraph_cluster;
    m_ng_backend->call(ng_function, outputs, ng_inputs);
    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute call done for cluster " << m_ngraph_cluster;
  }

 private:
  Graph m_graph;
  std::unordered_map<std::string, std::shared_ptr<ngraph::Function>>
      m_ng_functions;
  int m_ngraph_cluster;
  static std::shared_ptr<ng::runtime::Backend> m_ng_backend;
};
std::shared_ptr<ng::runtime::Backend> NGraphEncapsulateOp::m_ng_backend;

}  // namespace ngraph_bridge

REGISTER_KERNEL_BUILDER(Name("NGraphEncapsulate").Device(DEVICE_CPU),
                        ngraph_bridge::NGraphEncapsulateOp);

}  // namespace tensorflow
