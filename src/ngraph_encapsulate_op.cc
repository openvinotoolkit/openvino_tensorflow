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
#include <cstdlib>
#include <fstream>
#include <utility>

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
#include "ngraph_freshness_tracker.h"
#include "ngraph_log.h"
#include "ngraph_utils.h"

#include "ngraph/runtime/interpreter/int_backend.hpp"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

// For each I/O tensor, cache TF's data ptr and nGraph's TensorView
using NgFunctionIOCache = std::unordered_map<
    std::shared_ptr<ngraph::Function>,
    std::vector<std::pair<void*, shared_ptr<ng::runtime::TensorView>>>>;

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
      : OpKernel(ctx),
        m_graph(OpRegistry::Global()),
        m_freshness_tracker(nullptr) {
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
      NGRAPH_VLOG(2) << "Using INTERPRETER backend since "
                        "NGRAPH_EMBEDDED_IN_TENSORFLOW is enabled";
      m_ng_backend_name = "INTERPRETER";
#else
      const char* ng_backend_env_value = std::getenv("NGRAPH_TF_BACKEND");
      if (ng_backend_env_value != nullptr) {
        m_ng_backend_name = std::string(ng_backend_env_value);
        if (m_ng_backend_name.empty()) {
          m_ng_backend_name = "CPU";
        }
      } else {
        m_ng_backend_name = "CPU";
      }
#endif
      m_ng_backend = ng::runtime::Backend::create(m_ng_backend_name);
      OP_REQUIRES(ctx, m_ng_backend != nullptr,
                  errors::InvalidArgument("Cannot create nGraph backend"));
    }
  }

  ~NGraphEncapsulateOp() override {
    // If the kernel goes away, we must de-register all of its cached
    // functions
    // from the freshness tracker.
    if (m_freshness_tracker != nullptr) {
      for (auto kv : m_ng_functions) {
        m_freshness_tracker->RemoveUser(kv.second);
      }

      // TODO(amprocte): We should be able to unref the tracker here, but it
      // seems to screw things up in the C++ unit tests.
      // m_freshness_tracker->Unref();
    }
  }

  // TODO(amprocte): this needs to be made thread-safe (compilation cache OK?).
  void Compute(OpKernelContext* ctx) override {
    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute starting for cluster "
                   << m_ngraph_cluster;

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

    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute got inputs for cluster "
                   << m_ngraph_cluster;

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
            "tf_function_" + ctx->op_kernel().name() + ".json";
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

    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute got graph for cluster "
                   << m_ngraph_cluster;

    if (m_freshness_tracker == nullptr) {
      auto creator = [](NGraphFreshnessTracker** tracker) {
        *tracker = new NGraphFreshnessTracker();
        return Status::OK();
      };
      OP_REQUIRES_OK(
          ctx, ctx->resource_manager()->LookupOrCreate<NGraphFreshnessTracker>(
                   ctx->resource_manager()->default_container(),
                   "ngraph_freshness_tracker", &m_freshness_tracker, creator));
    }

    NGRAPH_VLOG(4)
        << "NGraphEncapsulateOp::Compute got freshness tracker for cluster "
        << m_ngraph_cluster;

    // Allocate tensors for arguments.
    vector<shared_ptr<ng::runtime::TensorView>> ng_inputs;

    std::vector<std::pair<void*, std::shared_ptr<ng::runtime::TensorView>>>&
        input_caches = m_ng_function_input_cache_map[ng_function];
    input_caches.resize(input_shapes.size());

    for (int i = 0; i < input_shapes.size(); i++) {
      ng::Shape ng_shape(input_shapes[i].dims());
      for (int j = 0; j < input_shapes[i].dims(); ++j) {
        ng_shape[j] = input_shapes[i].dim_size(j);
      }
      ng::element::Type ng_element_type;
      OP_REQUIRES_OK(ctx, TFDataTypeToNGraphElementType(ctx->input(i).dtype(),
                                                        &ng_element_type));

      // At the first call of the ng_function, both last_src_ptr and
      // last_tv shall point to null. Otherwise, they are retrived
      // from cache.
      void* last_src_ptr = input_caches[i].first;
      std::shared_ptr<ng::runtime::TensorView> last_tv = input_caches[i].second;

      void* current_src_ptr = (void*)DMAHelper::base(&ctx->input(i));
      std::shared_ptr<ng::runtime::TensorView> current_tv;

      if (m_ng_backend_name == "CPU") {
        // We need to check last_tv != nullptr, since there are cases where at
        // the first call to the ng_function, both the current_src_ptr (when the
        // input is a 0-sized tensor) and last_src_ptr (uninitialized at the
        // first call) are nullptr
        if (current_src_ptr == last_src_ptr && last_tv != nullptr) {
          // Mark each tensor as non-stale if:
          //   1. the freshness tracker says the tensor has not changed since
          //      the last time ng_function was called, and
          //   2. we are using the same tensor in this argument position as
          //      the one we used last time ng_function was called.
          if (m_freshness_tracker->IsFresh(current_src_ptr, ng_function)) {
            last_tv->set_stale(false);
          } else {
            last_tv->set_stale(true);
          }
          current_tv = last_tv;
        } else {
          current_tv = m_ng_backend->create_tensor(ng_element_type, ng_shape,
                                                   current_src_ptr);
          current_tv->set_stale(true);
        }
      } else {
        if (last_tv != nullptr) {
          current_tv = last_tv;
        } else {
          current_tv = m_ng_backend->create_tensor(ng_element_type, ng_shape);
        }
        current_tv->write(current_src_ptr, 0, current_tv->get_element_count() *
                                                  ng_element_type.size());
      }  // if (m_ng_backend_name == "CPU")

      input_caches[i] = std::make_pair(current_src_ptr, current_tv);
      ng_inputs.push_back(current_tv);
    }  // for (int i = 0; i < input_shapes.size(); i++)

    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute allocated argument tensors "
                      "for cluster "
                   << m_ngraph_cluster;

    // Allocate tensors for the results.
    vector<shared_ptr<ng::runtime::TensorView>> ng_outputs;

    std::vector<std::pair<void*, std::shared_ptr<ng::runtime::TensorView>>>&
        output_caches = m_ng_function_output_cache_map[ng_function];
    output_caches.resize(ng_function->get_output_size());

    for (auto i = 0; i < ng_function->get_output_size(); i++) {
      auto ng_shape = ng_function->get_output_shape(i);
      auto ng_element_type = ng_function->get_output_element_type(i);

      // Create the TF output tensor
      vector<int64> dims;
      for (auto dim : ng_shape) {
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
          ctx, ng_element_type == expected_elem_type,
          errors::Internal("Element type inferred by nGraph does not match "
                           "the element type expected by TensorFlow"));

      void* last_dst_ptr = output_caches[i].first;
      std::shared_ptr<ng::runtime::TensorView> last_tv =
          output_caches[i].second;

      void* current_dst_ptr = DMAHelper::base(output_tensor);
      std::shared_ptr<ng::runtime::TensorView> current_tv;

      if (m_ng_backend_name == "CPU") {
        // We need to check last_tv != nullptr, since there are cases where at
        // the first call to the ng_function, both the current_dst_ptr (when the
        // output is a 0-sized tensor) and last_dst_ptr (uninitialized at the
        // first call) are nullptr
        if (current_dst_ptr == last_dst_ptr && last_tv != nullptr) {
          current_tv = last_tv;
        } else {
          current_tv = m_ng_backend->create_tensor(ng_element_type, ng_shape,
                                                   current_dst_ptr);
        }
      } else {
        if (last_tv != nullptr) {
          current_tv = last_tv;
        } else {
          current_tv = m_ng_backend->create_tensor(ng_element_type, ng_shape);
        }
      }  // if (m_ng_backend_name == "CPU")

      current_tv->set_stale(true);
      output_caches[i] = std::make_pair(current_dst_ptr, current_tv);
      ng_outputs.push_back(current_tv);
    }  // for (auto i = 0; i < ng_function->get_output_size(); i++)

    NGRAPH_VLOG(4)
        << "NGraphEncapsulateOp::Compute allocated result tensors for cluster "
        << m_ngraph_cluster;

    // Execute the nGraph function.
    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute call starting for cluster "
                   << m_ngraph_cluster;
    m_ng_backend->call(ng_function, ng_outputs, ng_inputs);
    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute call done for cluster "
                   << m_ngraph_cluster;

    // Copy value to host if backend is not CPU
    if (m_ng_backend_name != "CPU") {
      for (size_t i = 0; i < output_caches.size(); ++i) {
        void* dst_ptr;
        std::shared_ptr<ng::runtime::TensorView> dst_tv;
        std::tie(dst_ptr, dst_tv) = output_caches[i];
        auto ng_element_type = dst_tv->get_tensor().get_element_type();
        dst_tv->read(dst_ptr, 0,
                     dst_tv->get_element_count() * ng_element_type.size());
      }
    }

    // Mark input tensors as fresh for the next time around.
    for (int i = 0; i < input_shapes.size(); i++) {
      void* src_ptr = (void*)DMAHelper::base(&ctx->input(i));
      m_freshness_tracker->MarkFresh(src_ptr, ng_function);
    }

    NGRAPH_VLOG(4)
        << "NGraphEncapsulateOp::Compute done marking fresh for cluster "
        << m_ngraph_cluster;
  }  // void Compute(OpKernelContext* ctx) override

 private:
  Graph m_graph;
  std::unordered_map<std::string, std::shared_ptr<ngraph::Function>>
      m_ng_functions;
  NgFunctionIOCache m_ng_function_input_cache_map;
  NgFunctionIOCache m_ng_function_output_cache_map;
  NGraphFreshnessTracker* m_freshness_tracker;
  int m_ngraph_cluster;
  static std::shared_ptr<ng::runtime::Backend> m_ng_backend;
  static std::string m_ng_backend_name;
};

std::shared_ptr<ng::runtime::Backend> NGraphEncapsulateOp::m_ng_backend;
std::string NGraphEncapsulateOp::m_ng_backend_name;

}  // namespace ngraph_bridge

REGISTER_KERNEL_BUILDER(Name("NGraphEncapsulate").Device(DEVICE_CPU),
                        ngraph_bridge::NGraphEncapsulateOp);

}  // namespace tensorflow
