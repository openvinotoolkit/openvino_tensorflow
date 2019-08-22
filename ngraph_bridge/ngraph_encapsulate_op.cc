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

#include "ngraph/event_tracing.hpp"
#include "ngraph/runtime/backend.hpp"

#if defined NGRAPH_DISTRIBUTED
#include "ngraph/distributed.hpp"
#endif

#include "logging/ngraph_log.h"
#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/ngraph_builder.h"
#include "ngraph_bridge/ngraph_cluster_manager.h"
#include "ngraph_bridge/ngraph_encapsulate_impl.h"
#include "ngraph_bridge/ngraph_encapsulate_op.h"
#include "ngraph_bridge/ngraph_freshness_tracker.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "ngraph_bridge/ngraph_pipelined_tensors.h"
#include "ngraph_bridge/ngraph_timer.h"
#include "ngraph_bridge/ngraph_utils.h"

#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
#include "ngraph_bridge/enable_variable_ops/ngraph_catalog.h"
#include "ngraph_bridge/enable_variable_ops/ngraph_var.h"
#endif

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

//---------------------------------------------------------------------------
//  NGraphEncapsulateOp::ctor
//---------------------------------------------------------------------------
NGraphEncapsulateOp::NGraphEncapsulateOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  ng_encap_impl.SetName(name());

  std::ostringstream oss;
  oss << "Encapsulate_" << ng_encap_impl.GetInstanceId() << ": " << name();

  ngraph::Event event(oss.str(), name(), "");

  NGRAPH_VLOG(1) << "NGraphEncapsulateOp: " << ng_encap_impl.GetInstanceId()
                 << " Name: " << name();

  GraphDef* graph_def;

  int cluster{-1};
  OP_REQUIRES_OK(ctx, ctx->GetAttr<int>("ngraph_cluster", &cluster));
  ng_encap_impl.SetNgraphCluster(cluster);
  graph_def =
      NGraphClusterManager::GetClusterGraph(ng_encap_impl.GetNgraphCluster());

  if (graph_def == nullptr) {
    string flib_key =
        "ngraph_cluster_" + to_string(ng_encap_impl.GetNgraphCluster());
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
    FunctionDefToBodyHelper(*fdef, {}, &flib, get_func_sig, &fnbody);
    CopyGraph(*fnbody->graph, &ng_encap_impl.m_graph);
  } else {
    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    OP_REQUIRES_OK(
        ctx, ConvertGraphDefToGraph(opts, *graph_def, &ng_encap_impl.m_graph));
  }

  int graph_id{-1};
  OP_REQUIRES_OK(ctx, ctx->GetAttr("ngraph_graph_id", &graph_id));
  ng_encap_impl.SetGraphId(graph_id);
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

  for (auto node : ng_encap_impl.m_graph.nodes()) {
    if (node->type_string() == "_Arg") {
      arg_nodes.push_back(node);

      int32 index;
      OP_REQUIRES_OK(ctx, GetNodeAttr(node->attrs(), "index", &index));
      if (index > max_arg_index) max_arg_index = index;
    }
  }

  int size = max_arg_index + 1;
  ng_encap_impl.ResizeStaticInputVector(size);

  for (int i = 0; i < size; i++) {
    ng_encap_impl.SetStaticInputVector(i, false);
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
    ng_encap_impl.SetStaticInputVector(index, is_static);
  }

  // Set the backend type for the op
  std::string backend_name;
  OP_REQUIRES_OK(ctx, ctx->GetAttr<string>("ngraph_backend", &backend_name));
  std::string device_id;
  OP_REQUIRES_OK(ctx, ctx->GetAttr<string>("ngraph_device_id", &device_id));
  // Get the optional attributes
  std::unordered_map<std::string, std::string> additional_attribute_map;
  auto node_def = ctx->def();
  auto additional_attributes = node_def.attr();
  for (auto itx : additional_attributes) {
    // Find the optional attributes to be sent to the backend.
    // The optional attributes have '_ngraph_' appended to the start
    // so we need to get rid of that and only send the remaining string
    // since the backend will only look for that.
    // '_ngraph_' is only appended for the bridge.
    // For e.g. _ngraph_ice_cores --> ice_cores
    if (itx.first.find("_ngraph_") != std::string::npos) {
      NGRAPH_VLOG(4) << "Attribute: " << itx.first.substr(strlen("_ngraph_"))
                     << " Value: " << itx.second.s();
      additional_attribute_map.insert(
          {itx.first.substr(strlen("_ngraph_")), itx.second.s()});
    }
  }

  // Concatenate the backend_name:device_id
  try {
    string be_name =
        BackendManager::GetBackendCreationString(backend_name, device_id);
    ng_encap_impl.SetOpBackend(be_name);
  } catch (const std::exception& exp) {
    Status status = errors::Internal(
        "Caught exception while creating backend string ", exp.what(), "\n");
    OP_REQUIRES_OK(ctx, status);
  }
  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Create backend " << def().name();
  BackendManager::CreateBackend(ng_encap_impl.GetOpBackend());
  // SetConfig will be called for each EncapsulateOp
  BackendManager::SetConfig(ng_encap_impl.GetOpBackend(),
                            additional_attribute_map);

  ng_encap_impl.SetExecCanCreateTensor(
      BackendManager::GetBackend(ng_encap_impl.GetOpBackend())
          ->executable_can_create_tensors());
  NGRAPH_VLOG(5) << "Executable can "
                 << (ng_encap_impl.GetExecCanCreateTensor() ? "" : "not")
                 << " create tensors";

  event.Stop();
  ngraph::Event::write_trace(event);
}

//---------------------------------------------------------------------------
//  ~NGraphEncapsulateOp()
//---------------------------------------------------------------------------
NGraphEncapsulateOp::~NGraphEncapsulateOp() {
  std::ostringstream oss;
  oss << "Destroy Encapsulate_" << ng_encap_impl.GetInstanceId() << ": "
      << name();
  ngraph::Event event(oss.str(), name(), "");
  NGRAPH_VLOG(2) << "~NGraphEncapsulateOp::" << name();
  // If the kernel goes away, we must de-register all of its cached
  // functions
  // from the freshness tracker.
  if (ng_encap_impl.GetNgraphFreshnessTracker() != nullptr) {
    for (auto kv : ng_encap_impl.GetNgExecMap()) {
      ng_encap_impl.GetNgraphFreshnessTracker()->RemoveUser(kv.second);
    }

    // TODO(amprocte): We should be able to unref the tracker here, but it
    // seems to screw things up in the C++ unit tests.
    // m_freshness_tracker->Unref();
  }

#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
  // Remove Entries from Catalog
  // Remove entries related to outputs
  for (int i = 0; i < ng_encap_impl.GetNumberOfOutputs(); i++) {
    string key =
        NGraphCatalog::CreateNodeKey(ng_encap_impl.GetGraphId(), name(), i);
    if (NGraphCatalog::ExistsInEncapOutputInfoMap(key)) {
      NGraphCatalog::DeleteFromEncapOutputInfoMap(key);
      NGRAPH_VLOG(2) << "Deleting from output info map " << key;
    }
  }

  NGRAPH_VLOG(2) << "Deleting from Output Copy Index map " << name();
  NGraphCatalog::DeleteFromEncapOutputCopyIndexesMap(ng_encap_impl.GetGraphId(),
                                                     name());

  // Remove entries related to inputs
  for (int i = 0; i < ng_encap_impl.GetNumberOfOutputs(); i++) {
    string key =
        NGraphCatalog::CreateNodeKey(ng_encap_impl.GetGraphId(), name(), i);
    if (NGraphCatalog::ExistsInInputVariableSharedNameMap(key)) {
      NGraphCatalog::DeleteFromInputVariableSharedNameMap(key);
      NGRAPH_VLOG(2) << "Deleting from input variable shared name map " << key;
    }
  }

#endif

  ng_encap_impl.ClearNgExecInputCache();
  ng_encap_impl.ClearNgExecOutputCache();
  ng_encap_impl.ClearNgExecMap();
  ng_encap_impl.ClearNgFunctionMap();
  ng_encap_impl.ClearNgExecPipelinedTensorMap();

  // Release the backend
  NGRAPH_VLOG(2) << "~NGraphEncapsulateOp():: ReleaseBackend";
  BackendManager::ReleaseBackend(ng_encap_impl.GetOpBackend());
  event.Stop();
  ngraph::Event::write_trace(event);
}

//---------------------------------------------------------------------------
// OpKernel::Compute
//---------------------------------------------------------------------------
void NGraphEncapsulateOp::Compute(OpKernelContext* ctx) {
  std::ostringstream oss;
  oss << "Execute: Encapsulate_" << ng_encap_impl.GetInstanceId() << ": "
      << name();
  ngraph::Event event(oss.str(), name(), "");

  Timer compute_time;
  std::lock_guard<std::mutex> lock(m_compute_lock);
  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute starting for cluster "
                 << ng_encap_impl.GetNgraphCluster();

  ngraph::Event event_func_maybe_create("FunctionMaybeCreate", name(), "");
  Timer function_lookup_or_create;

  std::vector<TensorShape> input_shapes;
  std::vector<const Tensor*> static_input_map;
  std::shared_ptr<ngraph::Function> ng_function;
  std::shared_ptr<ngraph::runtime::Executable> ng_exec;
  ng::runtime::Backend* op_backend;

  // TF input tensor
  std::vector<Tensor> tf_input_tensors;

  for (int i = 0; i < ctx->num_inputs(); i++) {
    tf_input_tensors.push_back(ctx->input(i));
  }

  int step_id = ctx->step_id();

  // Get ngraph executable and inputs information
  OP_REQUIRES_OK(ctx, ng_encap_impl.GetNgExecutable(
                          tf_input_tensors, input_shapes, static_input_map,
                          op_backend, ng_exec));

  NGRAPH_VLOG(1) << " Step_ID: " << step_id;
  NGRAPH_VLOG(4)
      << "NGraphEncapsulateOp::Compute got ngraph executable for cluster "
      << ng_encap_impl.GetNgraphCluster();

  int time_func_create_or_lookup = function_lookup_or_create.ElapsedInMS();
  event_func_maybe_create.Stop();

  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute got graph for cluster "
                 << ng_encap_impl.GetNgraphCluster();

  Timer create_or_lookup_tensors;

  int pipeline_idx = -1;
  PipelinedTensorVector inp_group_from_pipeline;
  PipelinedTensorVector out_group_from_pipeline;
  if (ng_encap_impl.GetExecCanCreateTensor()) {
    OP_REQUIRES_OK(ctx, ng_encap_impl.CachePipelinedTensorIfNeeded(ng_exec));
    // Cache must contain the ng_exec at this point

    try {
      std::tie(pipeline_idx, inp_group_from_pipeline, out_group_from_pipeline) =
          ng_encap_impl.GetTensorsFromPipeline(ng_exec);
    } catch (const std::exception& exp) {
      OP_REQUIRES(
          ctx, false,
          errors::Internal("Caught exception while getting pipelined tensors: ",
                           exp.what(), "\n"));
    }

    if (pipeline_idx < 0) {
      OP_REQUIRES(ctx, false,
                  errors::Internal("Expected GetTensorsFromPipeline to return "
                                   "an index >= 0, but got ",
                                   pipeline_idx));
    }
  }

  if (ng_encap_impl.GetNgraphFreshnessTracker() == nullptr) {
    auto creator = [](NGraphFreshnessTracker** tracker) {
      *tracker = new NGraphFreshnessTracker();
      return Status::OK();
    };
    NGraphFreshnessTracker* set_tracker = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->resource_manager()->LookupOrCreate<NGraphFreshnessTracker>(
                 ctx->resource_manager()->default_container(),
                 "ngraph_freshness_tracker", &set_tracker, creator));
    ng_encap_impl.SetNgraphFreshnessTracker(set_tracker);
  }

  NGRAPH_VLOG(4)
      << "NGraphEncapsulateOp::Compute got freshness tracker for cluster "
      << ng_encap_impl.GetNgraphCluster();

  // Allocate tensors for input arguments.
  ngraph::Event event_alloc_input("Input: maybe create", name(), "");

  vector<shared_ptr<ng::runtime::Tensor>> ng_inputs;
  int ng_input_tensor_size_in_bytes = 0;

  OP_REQUIRES_OK(ctx, ng_encap_impl.AllocateNGInputTensors(
                          tf_input_tensors, ng_exec, inp_group_from_pipeline,
                          op_backend, ng_inputs));

  event_alloc_input.Stop();

  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute allocated argument tensors "
                    "for cluster "
                 << ng_encap_impl.GetNgraphCluster();
  // Allocate tensors for the output results.
  ngraph::Event event_alloc_output("Output: maybe create", name(), "");
  vector<shared_ptr<ng::runtime::Tensor>> ng_outputs;
  int ng_output_tensor_size_in_bytes = 0;
  std::vector<Tensor*> tf_output_tensors;

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
    OP_REQUIRES_OK(ctx,
                   TFDataTypeToNGraphElementType(ctx->expected_output_dtype(i),
                                                 &expected_elem_type));
    OP_REQUIRES(
        ctx, ng_element_type == expected_elem_type,
        errors::Internal("Element type inferred by nGraph does not match "
                         "the element type expected by TensorFlow"));
  }

  OP_REQUIRES_OK(ctx, ng_encap_impl.AllocateNGOutputTensors(
                          tf_output_tensors, ng_exec, out_group_from_pipeline,
                          op_backend, ng_outputs));
  auto output_caches = ng_encap_impl.GetNgExecOutputCacheMap(ng_exec);

  event_alloc_output.Stop();
  NGRAPH_VLOG(4)
      << "NGraphEncapsulateOp::Compute allocated result tensors for cluster "
      << ng_encap_impl.GetNgraphCluster();

// Dealing with the output from Variable nodes here
#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute getting output variables "
                    "from resource manager "
                 << ng_encap_impl.GetNgraphCluster();

  ngraph::Event event_output_check_in_catalog(
      "Get Variable Outputs from Resource Manager", name(), "");

  for (auto i = 0; i < ng_exec->get_results().size(); i++) {
    void* current_dst_ptr = DMAHelper::base(tf_output_tensors[i]);
    std::shared_ptr<ng::runtime::Tensor> current_ng_tensor = nullptr;
    // if the output tensor is going to be assigned to a variable
    // we ask nGraph to provide the output directly in the variable tensor
    bool ref_exists = NGraphCatalog::ExistsInEncapOutputInfoMap(
        ng_encap_impl.GetGraphId(), name(), i);
    if (!ref_exists) {
      OP_REQUIRES(ctx, ng_outputs[i] != nullptr,
                  errors::Internal("Output ", i,
                                   " is not in Catalog nor was set from TF"));
      continue;
    }
    string output_key =
        NGraphCatalog::CreateNodeKey(ng_encap_impl.GetGraphId(), name(), i);
    string ref_var_name =
        NGraphCatalog::GetVariableSharedNameFromEncapOutputInfoMap(output_key);
    NGraphVar* var;
    OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup<NGraphVar>(
                            ctx->resource_manager()->default_container(),
                            ref_var_name, &var));
    current_ng_tensor = var->ng_tensor();

    // There might be scenarios where the input and output tensors are the
    // same.The staleness determined for the input tensor should be the
    // final staleness for the given tensor. The staleness of output
    // tensor should not matter as this tensor is meant to be
    // overwritten with the computed value.
    // So not setting staleness here.
    output_caches[i] = std::make_pair(current_dst_ptr, current_ng_tensor);
    var->Unref();
    ng_outputs[i] = current_ng_tensor;
  }
  event_output_check_in_catalog.Stop();
  ngraph::Event::write_trace(event_output_check_in_catalog);

  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute getting input variables "
                    "from resource manager "
                 << ng_encap_impl.GetNgraphCluster();

  ngraph::Event event_input_check_in_catalog(
      "Get Variable Inputs from Resource Manager", name(), "");

  // Dealing with the input from Variable nodes here
  for (int input_index = 0; input_index < input_shapes.size(); input_index++) {
    bool ref_exists = NGraphCatalog::ExistsInInputVariableSharedNameMap(
        ng_encap_impl.GetGraphId(), def().name(), input_index);

    if (!ref_exists) {
      OP_REQUIRES(ctx, ng_inputs[input_index] != nullptr,
                  errors::Internal("Input ", input_index,
                                   " is not in Catalog nor was set from TF"));
      continue;
    }

    string ref_var_name = NGraphCatalog::GetInputVariableSharedName(
        ng_encap_impl.GetGraphId(), def().name(), input_index);
    NGraphVar* var;
    OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup<NGraphVar>(
                            ctx->resource_manager()->default_container(),
                            ref_var_name, &var));

    if (var->sync_ng_tensor()) {
      int copies = ng_encap_impl.GetNumberOfCopies();
      ng_encap_impl.SetNumberOfCopies(copies++);
      stringstream str;
      str << "Var_Sync[" << input_index << "] ";
      ng_encap_impl.SetCopyLog(str.str());
    }

    void* current_tf_ptr = (void*)DMAHelper::base(&ctx->input(input_index));
    bool is_stale = !ng_encap_impl.GetNgraphFreshnessTracker()->IsFresh(
        current_tf_ptr, ng_exec);
    var->ng_tensor()->set_stale(is_stale);
    ng_inputs[input_index] = var->ng_tensor();

    var->Unref();
  }

  event_input_check_in_catalog.Stop();
  ngraph::Event::write_trace(event_input_check_in_catalog);
#endif

  int time_create_or_lookup_tensors = create_or_lookup_tensors.ElapsedInMS();

  // Execute the nGraph function.
  ngraph::Event event_execute_function("Execute nGraph", name(), "");
  Timer execute_function;
  {
    BackendManager::LockBackend(ng_encap_impl.GetOpBackend());
    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute call starting for cluster "
                   << ng_encap_impl.GetNgraphCluster();
    try {
      ng_exec->call(ng_outputs, ng_inputs);
    } catch (const std::exception& exp) {
      ng_function = ng_encap_impl.GetNgFunctionMap()[ng_exec];
      BackendManager::UnlockBackend(ng_encap_impl.GetOpBackend());
      NgraphSerialize("tf_function_error_" + ctx->op_kernel().name() + ".json",
                      ng_function);
      OP_REQUIRES(ctx, false,
                  errors::Internal(
                      "Caught exception while executing nGraph computation: ",
                      exp.what(), "\n"));
    } catch (...) {
      ng_function = ng_encap_impl.GetNgFunctionMap()[ng_exec];
      BackendManager::UnlockBackend(ng_encap_impl.GetOpBackend());
      NgraphSerialize("tf_function_error_" + ctx->op_kernel().name() + ".json",
                      ng_function);
      OP_REQUIRES(
          ctx, false,
          errors::Internal("Error in executing the nGraph computation\n"));
    }
    BackendManager::UnlockBackend(ng_encap_impl.GetOpBackend());
  }
  int time_execute_function = execute_function.ElapsedInMS();
  event_execute_function.Stop();

  long vm, rss;
  MemoryProfile(vm, rss);
  NGRAPH_VLOG(1) << "NGRAPH_TF_MEM_PROFILE:  OP_ID: "
                 << ng_encap_impl.GetInstanceId() << " Step_ID: " << step_id
                 << " Cluster: " << name() << " Input Tensors created: "
                 << ng_input_tensor_size_in_bytes / (1024 * 1024) << " MB"
                 << " Output Tensors created: "
                 << ng_output_tensor_size_in_bytes / (1024 * 1024) << " MB"
                 << " Total process memory: " << rss / (1024 * 1024) << " GB";

  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute call done for cluster "
                 << ng_encap_impl.GetNgraphCluster();

  // Copy value to host if backend is not CPU
  ngraph::Event event_copy_output("Output - copy back", name(), "");
  Timer copy_output_tensors_to_host;

  try {
    size_t output_tensor_count = output_caches.size();
    std::vector<std::unique_ptr<ngraph::Event>> output_copy_events;
#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
    if (ng_encap_impl.GetNumberOfOutputs() == -1) {
      NGRAPH_VLOG(4) << "Settig number of outputs for " << def().name();
      ng_encap_impl.SetNumberOfOutputs(ng_outputs.size());
      NGRAPH_VLOG(4) << "Setting number of inputs for " << def().name();
      ng_encap_impl.SetNumberOfInputs(ng_inputs.size());
    }
    for (size_t i = 0; i < output_tensor_count; ++i) {
      // Sync the Var Tensor if required
      string output_key = NGraphCatalog::CreateNodeKey(
          ng_encap_impl.GetGraphId(), def().name(), i);
      bool ref_exists = NGraphCatalog::ExistsInEncapOutputInfoMap(output_key);

      if (ref_exists) {
        NGRAPH_VLOG(4) << "Syncing the output var tensor " << output_key;

        // Get var
        string ref_var_name =
            NGraphCatalog::GetVariableSharedNameFromEncapOutputInfoMap(
                output_key);
        NGraphVar* var;
        OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup<NGraphVar>(
                                ctx->resource_manager()->default_container(),
                                ref_var_name, &var));

        if (NGraphCatalog::GetCopyToTFFromEncapOutputInfoMap(output_key)) {
          if (var->copy_ng_to_tf()) {
            int copies = ng_encap_impl.GetNumberOfCopies();
            ng_encap_impl.SetNumberOfCopies(copies++);
            ng_encap_impl.SetCopyLog(" COPY_TO_TF ");
          }
          if (!NGraphCatalog::GetIsTFJustLookingFromEncapOutputInfoMap(
                  output_key)) {
            // Some tf op might update the ng-tensor value so mark it stale
            ng_encap_impl.SetCopyLog(" SET_SYNC ");
            var->set_sync_ng_tensor(true);
          }
        }
        var->Unref();
      }

      std::shared_ptr<ng::runtime::Tensor> dst_ng_tensor;
      void* dst_ptr;
      std::tie(dst_ptr, dst_ng_tensor) = output_caches[i];

      if (ng_encap_impl.GetOpBackend() != "CPU" &&
          NGraphCatalog::EncapOutputIndexNeedsCopy(ng_encap_impl.GetGraphId(),
                                                   def().name(), i)) {
        int copies = ng_encap_impl.GetNumberOfCopies();
        ng_encap_impl.SetNumberOfCopies(copies++);
        stringstream log;
        log << " COPY_OP_VAL[" << i << "]";
        ng_encap_impl.SetCopyLog(log.str());

        NGRAPH_VLOG(4) << "Copying Output " << def().name() << " ,index: " << i;
        auto ng_element_type = dst_ng_tensor->get_element_type();
        size_t copy_size =
            dst_ng_tensor->get_element_count() * ng_element_type.size();
        string event_name =
            "Output_" + to_string(i) + "_" + to_string(copy_size);
        std::unique_ptr<ngraph::Event> event_copy_output_next(
            new ngraph::Event(event_name, name(), ""));
        dst_ng_tensor->read(dst_ptr, 0, dst_ng_tensor->get_element_count() *
                                            ng_element_type.size());
        event_copy_output_next->Stop();
        output_copy_events.push_back(std::move(event_copy_output_next));
      }
    }
#else
    if (ng_encap_impl.GetOpBackend() != "CPU") {
      for (size_t i = 0; i < output_tensor_count; ++i) {
        void* dst_ptr;
        std::shared_ptr<ng::runtime::Tensor> dst_ng_tensor;
        std::tie(dst_ptr, dst_ng_tensor) = output_caches[i];
        auto ng_element_type = dst_ng_tensor->get_element_type();
        std::unique_ptr<ngraph::Event> event_copy_output_next(new ngraph::Event(
            ("Output_" + std::to_string(i) + "_" +
             std::to_string(dst_ng_tensor->get_element_count() *
                            ng_element_type.size())),
            name(), ""));
        dst_ng_tensor->read(dst_ptr, 0, dst_ng_tensor->get_element_count() *
                                            ng_element_type.size());
        event_copy_output_next->Stop();
        output_copy_events.push_back(std::move(event_copy_output_next));
      }
    }
#endif
    // Now write the events back
    for (auto& next : output_copy_events) {
      ngraph::Event::write_trace(*next.get());
    }
  } catch (const std::exception& exp) {
    OP_REQUIRES(ctx, false,
                errors::Internal(
                    "Caught exception while transferring tensor data to host: ",
                    exp.what(), "\n"));
  } catch (...) {
    OP_REQUIRES(ctx, false, errors::Internal(
                                "Error in transferring tensor data to host\n"));
  }
  event_copy_output.Stop();

#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
  std::stringstream str;
  str << " Number of copies " << ng_encap_impl.GetNumberOfCopies() << "\n";
  ng_encap_impl.SetCopyLog(str.str());
  if (ng_encap_impl.GetLogCopies()) {
    cout << ng_encap_impl.GetCopyLog();
  }
#endif

  // Mark input tensors as fresh for the next time around.
  // Note: these ng_tensors are being marked fresh so that in the next
  // iteration if this encapsulate finds the tensor fresh, then it will use it
  for (int i = 0; i < input_shapes.size(); i++) {
    void* src_ptr = (void*)DMAHelper::base(&ctx->input(i));
    ng_encap_impl.GetNgraphFreshnessTracker()->MarkFresh(src_ptr, ng_exec);
  }
  int time_copy_output_tensors_to_host =
      copy_output_tensors_to_host.ElapsedInMS();

  if (ng_encap_impl.GetExecCanCreateTensor()) {
    try {
      ng_encap_impl.ReturnPipelinedTensors(ng_exec, pipeline_idx);
    } catch (const std::exception& exp) {
      OP_REQUIRES(ctx, false,
                  errors::Internal(
                      "Caught exception while returning pipelined tensors: ",
                      exp.what(), "\n"));
    }
  }

  NGRAPH_VLOG(4)
      << "NGraphEncapsulateOp::Compute done marking fresh for cluster "
      << ng_encap_impl.GetNgraphCluster();
  NGRAPH_VLOG(1) << "NGRAPH_TF_TIMING_PROFILE: OP_ID: "
                 << ng_encap_impl.GetInstanceId() << " Step_ID: " << step_id
                 << " Cluster: " << name()
                 << " Time-Compute: " << compute_time.ElapsedInMS()
                 << " Function-Create-or-Lookup: " << time_func_create_or_lookup
                 << " Create-and-copy-tensors: "
                 << time_create_or_lookup_tensors
                 << " Execute: " << time_execute_function
                 << " Copy-outputs-to-host: "
                 << time_copy_output_tensors_to_host;
  event.Stop();
  ngraph::Event::write_trace(event_func_maybe_create);
  ngraph::Event::write_trace(event_alloc_output);
  ngraph::Event::write_trace(event_alloc_input);
  ngraph::Event::write_trace(event_execute_function);
  ngraph::Event::write_trace(event_copy_output);
  ngraph::Event::write_trace(event);

}  // end compute

int NGraphEncapsulateImpl::s_instance_count = 0;

}  // namespace ngraph_bridge

REGISTER_KERNEL_BUILDER(Name("NGraphEncapsulate").Device(DEVICE_CPU),
                        ngraph_bridge::NGraphEncapsulateOp);

}  // namespace tensorflow
