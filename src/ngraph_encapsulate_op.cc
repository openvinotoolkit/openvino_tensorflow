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
#include "ngraph_mark_for_clustering.h"
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

    OP_REQUIRES_OK(ctx, ctx->GetAttr<int>("ngraph_cluster", &m_ngraph_cluster));
    graph_def = NGraphClusterManager::GetClusterGraph(m_ngraph_cluster);

    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    OP_REQUIRES_OK(ctx, ConvertGraphDefToGraph(opts, *graph_def, &m_graph));

    //
    // Initialize the "m_input_is_static" vector as follows:
    // (1) create m_input_is_static with n+1 elements, where n is the max arg
    //     index
    // (2) for each _Arg node n, set m_input_is_static[n.index] to true if n
    //     is driving any static input; else set it to false.
    //

    // Create the vector.
    int32 max_arg_index = -1;
    std::vector<const Node*> arg_nodes;

    for (auto node : m_graph.nodes()) {
      if (node->type_string() == "_Arg") {
        arg_nodes.push_back(node);

        int32 index;
        OP_REQUIRES_OK(ctx, GetNodeAttr(node->attrs(), "index", &index));
        if (index > max_arg_index) max_arg_index = index;
      }
    }

    m_input_is_static = std::vector<bool>(max_arg_index + 1, false);

    // Fill the vector.
    for (auto node : arg_nodes) {
      int32 index;
      OP_REQUIRES_OK(ctx, GetNodeAttr(node->attrs(), "index", &index));

      bool is_static = false;
      for (auto edge : node->out_edges()) {
        if (edge->IsControlEdge() || !edge->dst()->IsOp()) {
          continue;
        }

        NGRAPH_VLOG(5) << "For arg " << index << " checking edge "
                       << edge->DebugString();

        if (InputIsStatic(edge->dst(), edge->dst_input())) {
          NGRAPH_VLOG(5) << "Marking edge static: " << edge->DebugString();
          is_static = true;
          break;
        }
      }

      NGRAPH_VLOG(5) << "Marking arg " << index << " is_static: " << is_static;
      m_input_is_static[index] = is_static;
    }

    // Create the backend
    mutex_lock l(s_ng_backend_mutex);
    if (auto ptr = s_ng_backend_wptr.lock()) {
      m_ng_backend = ptr;
      return;
    }
    const char* ng_backend_env_value = std::getenv("NGRAPH_TF_BACKEND");
    if (ng_backend_env_value != nullptr) {
      s_ng_backend_name = std::string(ng_backend_env_value);
      if (s_ng_backend_name.empty()) {
        s_ng_backend_name = "CPU";
      }
    } else {
      s_ng_backend_name = "CPU";
    }
    m_ng_backend = ng::runtime::Backend::create(s_ng_backend_name);
    OP_REQUIRES(ctx, m_ng_backend != nullptr,
                errors::InvalidArgument("Cannot create nGraph backend"));
    s_ng_backend_wptr = m_ng_backend;
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

  template <typename T>
  static void TensorDataToStream(std::ostream& ostream, int64 n_elements,
                                 const char* data) {
    const T* data_T = reinterpret_cast<const T*>(data);
    for (size_t i = 0; i < n_elements; i++) {
      ostream << data_T[i] << ",";
    }
  }

  static Status TensorToStream(std::ostream& ostream, const Tensor& tensor) {
    const char* data = tensor.tensor_data().data();
    int64 n_elements = tensor.NumElements();
    switch (tensor.dtype()) {
      case DT_HALF:
        TensorDataToStream<Eigen::half>(ostream, n_elements, data);
        break;
      case DT_FLOAT:
        TensorDataToStream<float>(ostream, n_elements, data);
        break;
      case DT_DOUBLE:
        TensorDataToStream<double>(ostream, n_elements, data);
        break;
      case DT_UINT32:
        TensorDataToStream<uint32>(ostream, n_elements, data);
        break;
      case DT_INT32:
        TensorDataToStream<int32>(ostream, n_elements, data);
        break;
      case DT_UINT8:
      case DT_QUINT8:
        TensorDataToStream<uint8>(ostream, n_elements, data);
        break;
      case DT_UINT16:
      case DT_QUINT16:
        TensorDataToStream<uint16>(ostream, n_elements, data);
        break;
      case DT_INT8:
      case DT_QINT8:
        TensorDataToStream<int8>(ostream, n_elements, data);
        break;
      case DT_INT16:
      case DT_QINT16:
        TensorDataToStream<int16>(ostream, n_elements, data);
        break;
      case DT_UINT64:
        TensorDataToStream<uint64>(ostream, n_elements, data);
        break;
      case DT_INT64:
        TensorDataToStream<int64>(ostream, n_elements, data);
        break;
      case DT_BOOL:
        TensorDataToStream<bool>(ostream, n_elements, data);
        break;
      default:
        return errors::Internal("TensorToStream got unsupported data type ",
                                DataType_Name(tensor.dtype()));
        break;
    }
    return Status::OK();
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

    signature_ss << "/";

    std::vector<const Tensor*> static_input_map(ctx->num_inputs());
    for (int i = 0; i < ctx->num_inputs(); i++) {
      const Tensor& input_tensor = ctx->input(i);
      if (m_input_is_static[i]) {
        static_input_map[i] = &input_tensor;
        OP_REQUIRES_OK(ctx, TensorToStream(signature_ss, input_tensor));
        signature_ss << ";";
      }
    }

    std::shared_ptr<ngraph::Function> ng_function;
    std::string signature = signature_ss.str();

    if (NGRAPH_VLOG_IS_ON(5)) {
      NGRAPH_VLOG(5) << "Computed signature: " << signature;
    }

    auto it = m_ng_functions.find(signature);

    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute got inputs for cluster "
                   << m_ngraph_cluster;

    // Compile the graph using nGraph.
    //
    // TODO(amprocte): Investigate performance of the compilation cache.
    if (it == m_ng_functions.end()) {
      NGRAPH_VLOG(1) << "Compilation cache miss: " << ctx->op_kernel().name();
      OP_REQUIRES_OK(
          ctx, Builder::TranslateGraph(input_shapes, static_input_map, &m_graph,
                                       ng_function));

      // Serialize to nGraph if needed
      if (std::getenv("NGRAPH_ENABLE_SERIALIZE") != nullptr) {
        std::string file_name =
            "tf_function_" + ctx->op_kernel().name() + ".json";
        NGRAPH_VLOG(0) << "Serializing graph to: " << file_name;
        std::string js = ngraph::serialize(ng_function, 4);
        std::ofstream f;
        f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        try {
          f.open(file_name);
          f << js;
          f.close();
        } catch (std::ofstream::failure& e) {
          std::cerr << "Exception opening/closing file " << file_name << endl;
          std::cerr << e.what() << endl;
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

      try {
        if (s_ng_backend_name == "CPU") {
          // We need to check last_tv != nullptr, since there are cases where at
          // the first call to the ng_function, both the current_src_ptr (when
          // the input is a 0-sized tensor) and last_src_ptr (uninitialized at
          // the first call) are nullptr
          if (current_src_ptr == last_src_ptr && last_tv != nullptr) {
            // Mark each tensor as non-stale if:
            //   1. the freshness tracker says the tensor has not changed since
            //      the last time ng_function was called, and
            //   2. we are using the same tensor in this argument position as
            //      the one we used last time ng_function was called.
            last_tv->set_stale(
                !m_freshness_tracker->IsFresh(current_src_ptr, ng_function));
            current_tv = last_tv;
          } else {
            current_tv = m_ng_backend->create_tensor(ng_element_type, ng_shape,
                                                     current_src_ptr);
            current_tv->set_stale(true);
          }
        } else {
          if (last_tv != nullptr) {
            if (current_src_ptr == last_src_ptr) {
              last_tv->set_stale(
                  !m_freshness_tracker->IsFresh(current_src_ptr, ng_function));
            } else {
              last_tv->set_stale(true);
            }
            current_tv = last_tv;
          } else {
            current_tv = m_ng_backend->create_tensor(ng_element_type, ng_shape);
            current_tv->set_stale(true);
          }
          if (current_tv->get_stale()) {
            current_tv->write(
                current_src_ptr, 0,
                current_tv->get_element_count() * ng_element_type.size());
          }
        }  // if (s_ng_backend_name == "CPU")
      } catch (const std::exception& exp) {
        OP_REQUIRES(
            ctx, false,
            errors::Internal(
                "Caught exception while transferring tensor data to nGraph: ",
                exp.what(), "\n"));
      } catch (...) {
        OP_REQUIRES(
            ctx, false,
            errors::Internal("Error in transferring tensor data to nGraph\n"));
      }
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

      if (s_ng_backend_name == "CPU") {
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
      }  // if (s_ng_backend_name == "CPU")

      current_tv->set_stale(true);
      output_caches[i] = std::make_pair(current_dst_ptr, current_tv);
      ng_outputs.push_back(current_tv);
    }

    NGRAPH_VLOG(4)
        << "NGraphEncapsulateOp::Compute allocated result tensors for cluster "
        << m_ngraph_cluster;

    // Execute the nGraph function.
    {
      mutex_lock l(s_ng_backend_mutex);
      NGRAPH_VLOG(4)
          << "NGraphEncapsulateOp::Compute call starting for cluster "
          << m_ngraph_cluster;
      try {
        m_ng_backend->call(ng_function, ng_outputs, ng_inputs);
      } catch (const std::exception& exp) {
        OP_REQUIRES(ctx, false,
                    errors::Internal(
                        "Caught exception while executing nGraph computation: ",
                        exp.what(), "\n"));
      } catch (...) {
        OP_REQUIRES(
            ctx, false,
            errors::Internal("Error in executing the nGraph computation\n"));
      }
    }
    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute call done for cluster "
                   << m_ngraph_cluster;

    // Copy value to host if backend is not CPU
    try {
      if (s_ng_backend_name != "CPU") {
        for (size_t i = 0; i < output_caches.size(); ++i) {
          void* dst_ptr;
          std::shared_ptr<ng::runtime::TensorView> dst_tv;
          std::tie(dst_ptr, dst_tv) = output_caches[i];
          auto ng_element_type = dst_tv->get_tensor().get_element_type();
          dst_tv->read(dst_ptr, 0,
                       dst_tv->get_element_count() * ng_element_type.size());
        }
      }
    } catch (const std::exception& exp) {
      OP_REQUIRES(
          ctx, false,
          errors::Internal(
              "Caught exception while transferring tensor data to host: ",
              exp.what(), "\n"));
    } catch (...) {
      OP_REQUIRES(
          ctx, false,
          errors::Internal("Error in transferring tensor data to host\n"));
    }

    // Mark input tensors as fresh for the next time around.
    for (int i = 0; i < input_shapes.size(); i++) {
      void* src_ptr = (void*)DMAHelper::base(&ctx->input(i));
      m_freshness_tracker->MarkFresh(src_ptr, ng_function);
    }

    NGRAPH_VLOG(4)
        << "NGraphEncapsulateOp::Compute done marking fresh for cluster "
        << m_ngraph_cluster;
  }

 private:
  Graph m_graph;
  std::unordered_map<std::string, std::shared_ptr<ngraph::Function>>
      m_ng_functions;
  NgFunctionIOCache m_ng_function_input_cache_map;
  NgFunctionIOCache m_ng_function_output_cache_map;
  NGraphFreshnessTracker* m_freshness_tracker;
  int m_ngraph_cluster;
  std::vector<bool> m_input_is_static;

  static std::weak_ptr<ng::runtime::Backend> s_ng_backend_wptr;
  std::shared_ptr<ng::runtime::Backend> m_ng_backend
      GUARDED_BY(s_ng_backend_mutex);
  static std::string s_ng_backend_name;
  static mutex s_ng_backend_mutex;
};

std::weak_ptr<ng::runtime::Backend> NGraphEncapsulateOp::s_ng_backend_wptr;
std::string NGraphEncapsulateOp::s_ng_backend_name;
mutex NGraphEncapsulateOp::s_ng_backend_mutex;
}  // namespace ngraph_bridge

REGISTER_KERNEL_BUILDER(Name("NGraphEncapsulate").Device(DEVICE_CPU),
                        ngraph_bridge::NGraphEncapsulateOp);

}  // namespace tensorflow
