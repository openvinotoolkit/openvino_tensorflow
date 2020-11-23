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

namespace tensorflow {
namespace ngraph_bridge {

int NGraphEncapsulateOp::s_instance_id = 0;

//---------------------------------------------------------------------------
//  NGraphEncapsulateOp::ctor
//---------------------------------------------------------------------------
NGraphEncapsulateOp::NGraphEncapsulateOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  NGRAPH_VLOG(1) << "Create Executor " << name();
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
}

//---------------------------------------------------------------------------
// OpKernel::Compute
//---------------------------------------------------------------------------
void NGraphEncapsulateOp::Compute(OpKernelContext* ctx) {
  NGRAPH_VLOG(1) << "Compute using executor " << name();
  std::ostringstream oss;
  oss << "Execute: Encapsulate_" << ng_encap_impl_.GetInstanceId() << ": "
      << name();
  NG_TRACE(oss.str(), name(), "");

  Timer compute_time;
  std::lock_guard<std::mutex> lock(m_compute_lock_);
  int cluster_id = ng_encap_impl_.GetNgraphCluster();
  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute starting for cluster "
                 << cluster_id;
  int time_func_create_or_lookup;
  Timer function_lookup_or_create;

  std::vector<TensorShape> input_shapes;
  std::vector<const Tensor*> static_input_map;
  std::shared_ptr<Executable> ng_exec;

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
        << cluster_id;

    time_func_create_or_lookup = function_lookup_or_create.ElapsedInMS();
  }

  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute got graph for cluster "
                 << cluster_id;

  Timer create_or_lookup_tensors;

  // Allocate tensors for input arguments.
  vector<shared_ptr<ngraph::runtime::Tensor>> ng_inputs;
  int ng_input_tensor_size_in_bytes = 0;
  {
    NG_TRACE("Input: maybe create", name(), "");
    OP_REQUIRES_OK(
        ctx, ng_encap_impl_.AllocateNGTensors(tf_input_tensors, ng_inputs));
  }

  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute allocated argument tensors "
                    "for cluster "
                 << cluster_id;

  // Allocate tensors for the output results.

  auto results = ng_exec->get_results();
  std::vector<shared_ptr<ngraph::runtime::Tensor>> ng_outputs(results.size(),
                                                              nullptr);
  std::vector<int> dyn_shape_tensors;
  for (auto i = 0; i < results.size(); i++) {
    auto ng_element = results[i];
    if (ng_element->get_output_partial_shape(0).is_dynamic()) {
      NGRAPH_VLOG(4)
          << "NGraphEncapsulateOp::Compute skipping output allocation for "
             "dynamic tensor at index"
          << i;
      dyn_shape_tensors.push_back(i);
      continue;
    }

    // Create the TF output tensor
    auto ng_shape = ng_element->get_shape();
    TensorShape tf_shape;
    for (auto dim : ng_shape) {
      tf_shape.AddDim(dim);
    }
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(i, tf_shape, &output_tensor));

    // Make sure the nGraph-inferred element type agrees with what TensorFlow
    // expected.
    ngraph::element::Type expected_elem_type;
    auto ng_element_type = ng_element->get_element_type();
    OP_REQUIRES_OK(ctx,
                   TFDataTypeToNGraphElementType(ctx->expected_output_dtype(i),
                                                 &expected_elem_type));
    OP_REQUIRES(
        ctx, ng_element_type == expected_elem_type,
        errors::Internal("Element type inferred by nGraph does not match "
                         "the element type expected by TensorFlow"));
    ng_outputs[i] =
        make_shared<IETensor>(ng_element_type, ng_shape, output_tensor->data());
  }
  NGRAPH_VLOG(4)
      << "NGraphEncapsulateOp::Compute allocated result tensors for cluster "
      << cluster_id;

  int time_create_or_lookup_tensors = create_or_lookup_tensors.ElapsedInMS();
  // Execute the nGraph function.
  int time_execute_function;
  {
    NG_TRACE("Execute nGraph", name(), "");
    Timer execute_function;
    {
      NGRAPH_VLOG(4)
          << "NGraphEncapsulateOp::Compute call starting for cluster "
          << cluster_id;
      try {
        ng_exec->call(ng_inputs, ng_outputs);
      } catch (const std::exception& exp) {
        string status_string = "Caught exception while executing cluster " +
                               to_string(cluster_id) + string(exp.what());
        OP_REQUIRES(ctx, false, errors::Internal(status_string));
      } catch (...) {
        string status_string =
            "Caught exception while executing cluster " + to_string(cluster_id);
        OP_REQUIRES(ctx, false, errors::Internal(status_string));
      }
    }
    time_execute_function = execute_function.ElapsedInMS();
  }

  for (auto i : dyn_shape_tensors) {
    auto ng_output = ng_outputs[i];
    // Create the TF output tensor
    auto ng_shape = ng_output->get_shape();
    TensorShape tf_shape;
    for (auto dim : ng_shape) {
      tf_shape.AddDim(dim);
    }

    // Zero-copy IE tensor to TF
    IETensorBuffer* tf_buffer =
        new IETensorBuffer(static_pointer_cast<IETensor>(ng_output));
    Tensor tf_tensor(ctx->expected_output_dtype(i), tf_shape, tf_buffer);
    ctx->set_output(i, tf_tensor);
  }

  long vm, rss;
  MemoryProfile(vm, rss);
  NGRAPH_VLOG(1) << "NGRAPH_TF_MEM_PROFILE:  OP_ID: "
                 << ng_encap_impl_.GetInstanceId() << " Step_ID: " << step_id
                 << " Cluster: " << name() << " Input Tensors created: "
                 << ng_input_tensor_size_in_bytes / (1024 * 1024) << " MB"
                 << " Total process memory: " << rss / (1024 * 1024) << " GB";

  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute call done for cluster "
                 << ng_encap_impl_.GetNgraphCluster();

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
                 << " Execute: " << time_execute_function;
}  // end compute

int NGraphEncapsulateImpl::s_instance_count = 0;

}  // namespace ngraph_bridge

REGISTER_OP("_nGraphEncapsulate")
    .Input("args: Targuments")
    .Attr("Targuments: list(type) >= 0")
    .Output("results: Tresults")
    .Attr("Tresults: list(type) >= 0")
    .Attr("ngraph_cluster: int")
    .Attr("ngraph_graph_id: int")
    .SetIsStateful()
    .Doc("nGraph Encapsulation Op. For use by the nGraph JIT only.");

REGISTER_KERNEL_BUILDER(Name("_nGraphEncapsulate").Device(DEVICE_CPU),
                        ngraph_bridge::NGraphEncapsulateOp);

}  // namespace tensorflow
