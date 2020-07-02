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
#include <cstdlib>
#include <mutex>
#include <utility>

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"

#include "ngraph/runtime/backend.hpp"

#include "logging/ngraph_log.h"
#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/ngraph_builder.h"
#include "ngraph_bridge/ngraph_cluster_manager.h"
#include "ngraph_bridge/ngraph_encapsulate_impl.h"
#include "ngraph_bridge/ngraph_encapsulate_op.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "ngraph_bridge/ngraph_timer.h"
#include "ngraph_bridge/ngraph_utils.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

int NGraphEncapsulateOp::s_instance_id = 0;

//---------------------------------------------------------------------------
//  NGraphEncapsulateOp::ctor
//---------------------------------------------------------------------------
NGraphEncapsulateOp::NGraphEncapsulateOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  // Set the backend type for the this NGraphEncapsulate Op
  std::string backend_name;
  OP_REQUIRES_OK(ctx, ctx->GetAttr<string>("ngraph_backend", &backend_name));
  std::string device_id;
  OP_REQUIRES_OK(ctx, ctx->GetAttr<string>("ngraph_device_id", &device_id));

  // Concatenate the backend_name:device_id
  string be_name =
      BackendManager::GetBackendCreationString(backend_name, device_id);

  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Create backend " << def().name()
                 << "BE: " << be_name;
  OP_REQUIRES_OK(ctx, BackendManager::CreateBackend(be_name));

  auto backend = BackendManager::GetBackend(be_name);
  OP_REQUIRES(
      ctx, backend != nullptr,
      errors::Internal("Cannot get the backend object for BE: ", be_name));

  CreateLegacyExecutor(ctx, be_name);
}

//---------------------------------------------------------------------------
//  CreateLegacyExecutor
//---------------------------------------------------------------------------
void NGraphEncapsulateOp::CreateLegacyExecutor(OpKernelConstruction* ctx,
                                               const string& backend_name) {
  NGRAPH_VLOG(1) << "Create Legacy Executor " << name();
  ng_encap_impl_.SetName(name());

  std::ostringstream oss;
  oss << "Encapsulate_" << ng_encap_impl_.GetInstanceId() << ": " << name();

  NG_TRACE(oss.str(), name(), "");

  NGRAPH_VLOG(1) << "NGraphEncapsulateOp: " << ng_encap_impl_.GetInstanceId()
                 << " Name: " << name();

  GraphDef* graph_def;

  int cluster{-1};
  OP_REQUIRES_OK(ctx, ctx->GetAttr<int>("ngraph_cluster", &cluster));
  ng_encap_impl_.SetNgraphCluster(cluster);
  graph_def =
      NGraphClusterManager::GetClusterGraph(ng_encap_impl_.GetNgraphCluster());

  if (graph_def == nullptr) {
    string flib_key =
        "ngraph_cluster_" + to_string(ng_encap_impl_.GetNgraphCluster());
    // Read graphdef from function library
    const FunctionLibraryDefinition flib =
        *ctx->function_library()->GetFunctionLibraryDefinition();
    const FunctionDef* fdef = flib.Find(flib_key);
    OP_REQUIRES(
        ctx, fdef != nullptr,
        errors::Internal("Did not find graphdef for encapsulate ", flib_key,
                         " in NGraphClusterManager or function library"));
    // TODO: how to convert from functiondef to graphdef. Anything easier?
    std::unique_ptr<FunctionBody> fnbody;
    const auto get_func_sig = [&flib](const string& op, const OpDef** sig) {
      return flib.LookUpOpDef(op, sig);
    };
    Status status =
        FunctionDefToBodyHelper(*fdef, {}, &flib, get_func_sig, &fnbody);
    if (!status.ok()) {
      NGRAPH_VLOG(2) << "FunctionDefToBodyHelper returned a not ok status.";
    }
    CopyGraph(*fnbody->graph, &ng_encap_impl_.m_graph);
  } else {
    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    OP_REQUIRES_OK(
        ctx, ConvertGraphDefToGraph(opts, *graph_def, &ng_encap_impl_.m_graph));
  }

  int graph_id{-1};
  OP_REQUIRES_OK(ctx, ctx->GetAttr("ngraph_graph_id", &graph_id));
  ng_encap_impl_.SetGraphId(graph_id);
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

  for (auto node : ng_encap_impl_.m_graph.nodes()) {
    if (node->type_string() == "_Arg") {
      arg_nodes.push_back(node);

      int32 index;
      OP_REQUIRES_OK(ctx, GetNodeAttr(node->attrs(), "index", &index));
      if (index > max_arg_index) max_arg_index = index;
    }
  }

  int size = max_arg_index + 1;
  ng_encap_impl_.ResizeStaticInputVector(size);

  for (int i = 0; i < size; i++) {
    ng_encap_impl_.SetStaticInputVector(i, false);
  }

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
    ng_encap_impl_.SetStaticInputVector(index, is_static);
  }

  // Get the optional attributes
  std::unordered_map<std::string, std::string> additional_attribute_map;
  auto node_def = ctx->def();
  OP_REQUIRES_OK(ctx, ng_encap_impl_.ParseNodeAttributes(
                          node_def.attr(), &additional_attribute_map));

  ng_encap_impl_.SetOpBackend(backend_name);

  // SetConfig will be called for each EncapsulateOp
  BackendManager::SetConfig(ng_encap_impl_.GetOpBackend(),
                            additional_attribute_map);
}

//---------------------------------------------------------------------------
//  ~NGraphEncapsulateOp()
//---------------------------------------------------------------------------
NGraphEncapsulateOp::~NGraphEncapsulateOp() {
  std::ostringstream oss;
  oss << "Destroy Encapsulate_" << ng_encap_impl_.GetInstanceId() << ": "
      << name();
  NG_TRACE(oss.str(), name(), "");
  NGRAPH_VLOG(2) << "~NGraphEncapsulateOp::" << name();
  ng_encap_impl_.ClearExecMaps();

  // Release the backend
  NGRAPH_VLOG(2) << "~NGraphEncapsulateOp():: ReleaseBackend";
  BackendManager::ReleaseBackend(ng_encap_impl_.GetOpBackend());
}

//---------------------------------------------------------------------------
// OpKernel::Compute
//---------------------------------------------------------------------------
void NGraphEncapsulateOp::Compute(OpKernelContext* ctx) {
  NG_TRACE("NGEncap::Compute::" + name(), name(), "");
  NGRAPH_VLOG(1) << "NGraphEncapsulateOp::Compute: Using Legacy Executor";
  ComputeUsingLegacyExecutor(ctx);
}

//---------------------------------------------------------------------------
//    ComputeUsingLegacyExecutor
//---------------------------------------------------------------------------
void NGraphEncapsulateOp::ComputeUsingLegacyExecutor(OpKernelContext* ctx) {
  NGRAPH_VLOG(1) << "Compute using Legacy Executor " << name();
  std::ostringstream oss;
  oss << "Execute: Encapsulate_" << ng_encap_impl_.GetInstanceId() << ": "
      << name();
  NG_TRACE(oss.str(), name(), "");

  Timer compute_time;
  std::lock_guard<std::mutex> lock(m_compute_lock_);
  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute starting for cluster "
                 << ng_encap_impl_.GetNgraphCluster();
  int time_func_create_or_lookup;
  Timer function_lookup_or_create;

  std::vector<TensorShape> input_shapes;
  std::vector<const Tensor*> static_input_map;
  std::shared_ptr<ngraph::runtime::Executable> ng_exec;
  // TF input tensor
  std::vector<Tensor> tf_input_tensors;
  int step_id;
  {
    NG_TRACE("FunctionMaybeCreate", name(), "");
    for (int i = 0; i < ctx->num_inputs(); i++) {
      tf_input_tensors.push_back(ctx->input(i));
    }

    step_id = ctx->step_id();

    // Get ngraph executable and inputs information
    OP_REQUIRES_OK(
        ctx, ng_encap_impl_.GetNgExecutable(tf_input_tensors, input_shapes,
                                            static_input_map, ng_exec));

    NGRAPH_VLOG(1) << " Step_ID: " << step_id;
    NGRAPH_VLOG(4)
        << "NGraphEncapsulateOp::Compute got ngraph executable for cluster "
        << ng_encap_impl_.GetNgraphCluster();

    time_func_create_or_lookup = function_lookup_or_create.ElapsedInMS();
  }

  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute got graph for cluster "
                 << ng_encap_impl_.GetNgraphCluster();

  Timer create_or_lookup_tensors;

  // Allocate tensors for input arguments.
  vector<shared_ptr<ng::runtime::Tensor>> ng_inputs;
  int ng_input_tensor_size_in_bytes = 0;
  {
    NG_TRACE("Input: maybe create", name(), "");
    OP_REQUIRES_OK(ctx, ng_encap_impl_.AllocateNGInputTensors(
                            tf_input_tensors, ng_exec, ng_inputs));
  }

  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute allocated argument tensors "
                    "for cluster "
                 << ng_encap_impl_.GetNgraphCluster();
  // Allocate tensors for the output results.
  vector<shared_ptr<ng::runtime::Tensor>> ng_outputs;
  int ng_output_tensor_size_in_bytes = 0;
  std::vector<Tensor*> tf_output_tensors;
  std::vector<std::pair<void*, shared_ptr<ng::runtime::Tensor>>> output_caches;
  {
    NG_TRACE("Output: maybe create", name(), "");
    for (auto i = 0; i < ng_exec->get_results().size(); i++) {
      auto ng_element = ng_exec->get_results()[i];
      auto ng_shape = ng_element->get_shape();
      auto ng_element_type = ng_element->get_element_type();

      // Create the TF output tensor
      vector<int64> dims;
      for (auto dim : ng_shape) {
        dims.push_back(dim);
      }
      TensorShape tf_shape(dims);
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, tf_shape, &output_tensor));
      tf_output_tensors.push_back(output_tensor);

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
    }

    OP_REQUIRES_OK(ctx, ng_encap_impl_.AllocateNGOutputTensors(
                            tf_output_tensors, ng_exec, ng_outputs));
    output_caches = ng_encap_impl_.GetNgExecOutputCacheMap(ng_exec);
  }
  NGRAPH_VLOG(4)
      << "NGraphEncapsulateOp::Compute allocated result tensors for cluster "
      << ng_encap_impl_.GetNgraphCluster();

  int time_create_or_lookup_tensors = create_or_lookup_tensors.ElapsedInMS();

  // Execute the nGraph function.
  int time_execute_function;
  {
    NG_TRACE("Execute nGraph", name(), "");
    Timer execute_function;
    {
      BackendManager::LockBackend(ng_encap_impl_.GetOpBackend());
      NGRAPH_VLOG(4)
          << "NGraphEncapsulateOp::Compute call starting for cluster "
          << ng_encap_impl_.GetNgraphCluster();
      try {
        ng_exec->call(ng_outputs, ng_inputs);
      } catch (const std::exception& exp) {
        BackendManager::UnlockBackend(ng_encap_impl_.GetOpBackend());
        Status st = ng_encap_impl_.DumpNgFunction(
            "tf_function_error_" + ctx->op_kernel().name() + ".json", ng_exec);
        string status_string =
            "Caught exception while executing nGraph computation: " +
            string(exp.what()) +
            (st.ok() ? "" : (" Also error in dumping serialized function: " +
                             st.error_message()));
        OP_REQUIRES(ctx, false, errors::Internal(status_string));
      } catch (...) {
        BackendManager::UnlockBackend(ng_encap_impl_.GetOpBackend());
        Status st = ng_encap_impl_.DumpNgFunction(
            "tf_function_error_" + ctx->op_kernel().name() + ".json", ng_exec);
        string status_string =
            "Error in executing the nGraph computation." +
            (st.ok() ? "" : (" Also error in dumping serialized function: " +
                             st.error_message()));
        OP_REQUIRES(ctx, false, errors::Internal(status_string));
      }
      BackendManager::UnlockBackend(ng_encap_impl_.GetOpBackend());
    }
    time_execute_function = execute_function.ElapsedInMS();
  }

  long vm, rss;
  MemoryProfile(vm, rss);
  NGRAPH_VLOG(1) << "NGRAPH_TF_MEM_PROFILE:  OP_ID: "
                 << ng_encap_impl_.GetInstanceId() << " Step_ID: " << step_id
                 << " Cluster: " << name() << " Input Tensors created: "
                 << ng_input_tensor_size_in_bytes / (1024 * 1024) << " MB"
                 << " Output Tensors created: "
                 << ng_output_tensor_size_in_bytes / (1024 * 1024) << " MB"
                 << " Total process memory: " << rss / (1024 * 1024) << " GB";

  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute call done for cluster "
                 << ng_encap_impl_.GetNgraphCluster();

  // Copy value to host if backend is not CPU
  Timer copy_output_tensors_to_host;
  {
    NG_TRACE("Output - copy back", name(), "");
    try {
      size_t output_tensor_count = output_caches.size();
      if (ng_encap_impl_.GetOpBackend() != "CPU") {
        for (size_t i = 0; i < output_tensor_count; ++i) {
          void* dst_ptr;
          std::shared_ptr<ng::runtime::Tensor> dst_ng_tensor;
          std::tie(dst_ptr, dst_ng_tensor) = output_caches[i];
          auto ng_element_type = dst_ng_tensor->get_element_type();
          NG_TRACE(("Output_" + std::to_string(i) + "_" +
                    std::to_string(dst_ng_tensor->get_element_count() *
                                   ng_element_type.size())),
                   name(), "");
          dst_ng_tensor->read(dst_ptr, dst_ng_tensor->get_element_count() *
                                           ng_element_type.size());
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
  }

  int time_copy_output_tensors_to_host =
      copy_output_tensors_to_host.ElapsedInMS();

  NGRAPH_VLOG(4)
      << "NGraphEncapsulateOp::Compute done marking fresh for cluster "
      << ng_encap_impl_.GetNgraphCluster();
  NGRAPH_VLOG(1) << "NGRAPH_TF_TIMING_PROFILE: OP_ID: "
                 << ng_encap_impl_.GetInstanceId() << " Step_ID: " << step_id
                 << " Cluster: " << name()
                 << " Time-Compute: " << compute_time.ElapsedInMS()
                 << " Function-Create-or-Lookup: " << time_func_create_or_lookup
                 << " Create-and-copy-tensors: "
                 << time_create_or_lookup_tensors
                 << " Execute: " << time_execute_function
                 << " Copy-outputs-to-host: "
                 << time_copy_output_tensors_to_host;
}  // end compute

int NGraphEncapsulateImpl::s_instance_count = 0;

}  // namespace ngraph_bridge

REGISTER_KERNEL_BUILDER(Name("NGraphEncapsulate").Device(DEVICE_CPU),
                        ngraph_bridge::NGraphEncapsulateOp);

}  // namespace tensorflow
