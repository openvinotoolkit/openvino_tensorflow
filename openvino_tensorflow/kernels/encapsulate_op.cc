/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
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
#include "tensorflow/core/public/version.h"
#if (TF_MAJOR_VERSION >= 2) && (TF_MINOR_VERSION > 2)
#include "tensorflow/core/common_runtime/graph_constructor.h"
#else
#include "tensorflow/core/graph/graph_constructor.h"
#endif
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/public/session.h"

#include "logging/ovtf_log.h"
#include "openvino_tensorflow/backend_manager.h"
#include "openvino_tensorflow/cluster_manager.h"
#include "openvino_tensorflow/mark_for_clustering.h"
#include "openvino_tensorflow/ovtf_builder.h"
#include "openvino_tensorflow/ovtf_timer.h"
#include "openvino_tensorflow/ovtf_utils.h"

#ifdef _WIN32
#define EXPAND(x) x
#define TF_NEW_ID_FOR_INIT_2(m, c, ...) \
  EXPAND(m(c, __VA_ARGS__))  // L145 selective_registration.h
#define TF_EXTRACT_KERNEL_NAME_IMPL(m, ...) \
  EXPAND(m(__VA_ARGS__))  // L1431 op_kernel.h
#endif

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

class NGraphEncapsulateOp : public OpKernel {
 public:
  explicit NGraphEncapsulateOp(OpKernelConstruction* ctx);
  ~NGraphEncapsulateOp() override;
  void Compute(OpKernelContext* ctx) override;

 private:
  Status GetExecutable(const std::vector<Tensor>& tf_input_tensors,
                       std::shared_ptr<Executable>& ng_exec);
  Status Fallback(OpKernelContext* ctx);

  std::mutex m_compute_lock_;
  Graph m_graph;
  int m_cluster_id;
  int m_function_cache_depth_in_items = 16;
  string m_name;
  std::vector<bool> m_input_is_static;
  std::list<std::string> m_lru;
  std::unordered_map<std::string, std::shared_ptr<Executable>> m_ng_exec_map;
  ngraph::ResultVector zero_dim_outputs;
  std::shared_ptr<tensorflow::Session> m_session;
  std::vector<std::string> m_session_input_names;
  std::vector<std::string> m_session_output_names;
};

NGraphEncapsulateOp::NGraphEncapsulateOp(OpKernelConstruction* ctx)
    : OpKernel(ctx), m_graph(OpRegistry::Global()) {
  OVTF_VLOG(1) << "Create Executor " << name();
  m_name = name();

  OP_REQUIRES_OK(ctx, ctx->GetAttr<int>("ovtf_cluster", &m_cluster_id));
  std::ostringstream oss;
  oss << "Encapsulate_" << m_cluster_id << ": " << name();

  OVTF_VLOG(1) << "NGraphEncapsulateOp: " << m_cluster_id
               << " Name: " << name();

  GraphDef* graph_def = NGraphClusterManager::GetClusterGraph(m_cluster_id);
  if (graph_def == nullptr) {
    string flib_key = "ovtf_cluster_" + to_string(m_cluster_id);
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
      OVTF_VLOG(2) << "FunctionDefToBodyHelper returned a not ok status.";
    }
    CopyGraph(*fnbody->graph, &m_graph);
  } else {
    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    OP_REQUIRES_OK(ctx, ConvertGraphDefToGraph(opts, *graph_def, &m_graph));
  }

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

  int size = max_arg_index + 1;
  m_input_is_static.resize(size);

  for (int i = 0; i < size; i++) {
    m_input_is_static[i] = false;
  }

  for (auto node : arg_nodes) {
    int32 index;
    OP_REQUIRES_OK(ctx, GetNodeAttr(node->attrs(), "index", &index));

    bool is_static = false;
    for (auto edge : node->out_edges()) {
      if (edge->IsControlEdge() || !edge->dst()->IsOp()) {
        continue;
      }

      OVTF_VLOG(5) << "For arg " << index << " checking edge "
                   << edge->DebugString();

      if (InputIsStatic(edge->dst(), edge->dst_input())) {
        OVTF_VLOG(5) << "Marking edge static: " << edge->DebugString();
        is_static = true;
        break;
      }
    }
    OVTF_VLOG(5) << "Marking arg " << index << " is_static: " << is_static;
    m_input_is_static[index] = is_static;
  }
}

NGraphEncapsulateOp::~NGraphEncapsulateOp() {
  std::ostringstream oss;
  oss << "Destroy Encapsulate_" << m_cluster_id << ": " << name();
  OVTF_VLOG(2) << "~NGraphEncapsulateOp::" << name();
  NGraphClusterManager::SetMRUExecutable(m_cluster_id, nullptr);
  m_ng_exec_map.clear();
}

void NGraphEncapsulateOp::Compute(OpKernelContext* ctx) {
  OVTF_VLOG(1) << "Compute using executor " << name();
  std::ostringstream oss;
  oss << "Execute: Encapsulate_" << m_cluster_id << ": " << name();
  OVTF_VLOG(4) << "NGraphEncapsulateOp::Compute starting for cluster "
               << m_cluster_id;

  if (NGraphClusterManager::CheckClusterFallback(m_cluster_id)) {
    OP_REQUIRES_OK(ctx, Fallback(ctx));
    return;
  }

  Timer compute_time;
  std::lock_guard<std::mutex> lock(m_compute_lock_);
  int time_func_create_or_lookup;
  Timer function_lookup_or_create;

  bool multi_req_execution = false;
  if (std::getenv("OPENVINO_TF_ENABLE_BATCHING")) {
    OVTF_VLOG(2) << "Batching is enabled" << name();
    multi_req_execution = true;
  }

  // TF input tensor
  std::vector<Tensor> tf_input_tensors;
  std::shared_ptr<Executable> ng_exec;
  int step_id;
  {
    for (int i = 0; i < ctx->num_inputs(); i++) {
      tf_input_tensors.push_back(ctx->input(i));
    }

    step_id = ctx->step_id();

    // Get ngraph executable and inputs information
    Status getex_status = GetExecutable(tf_input_tensors, ng_exec);
    NGraphClusterManager::SetMRUExecutable(m_cluster_id, ng_exec);
    if (getex_status != Status::OK()) {
      if (NGraphClusterManager::IsClusterFallbackEnabled()) {
        OP_REQUIRES_OK(ctx, Fallback(ctx));
        return;
      } else {
        OP_REQUIRES_OK(ctx, getex_status);
      }
    }

    OVTF_VLOG(1) << " Step_ID: " << step_id;
    OVTF_VLOG(4)
        << "NGraphEncapsulateOp::Compute got ngraph executable for cluster "
        << m_cluster_id;

    time_func_create_or_lookup = function_lookup_or_create.ElapsedInMS();
  }

  OVTF_VLOG(4) << "NGraphEncapsulateOp::Compute got graph for cluster "
               << m_cluster_id;

  Timer create_or_lookup_tensors;
  vector<shared_ptr<ov::Tensor>> ng_inputs;
  ng_inputs.resize(tf_input_tensors.size());
  int ng_input_tensor_size_in_bytes = 0;
  {
    // Allocate tensors for input arguments.
    for (int i = 0; i < tf_input_tensors.size(); i++) {
      ov::Shape ng_shape(tf_input_tensors[i].shape().dims());
      for (int j = 0; j < tf_input_tensors[i].shape().dims(); ++j) {
        ng_shape[j] = tf_input_tensors[i].shape().dim_size(j);
      }
      auto check_ng_shape = [ng_shape]() {
        if (ng_shape.size() > 0) {
          for (auto dim : ng_shape) {
            if (dim == 0) return true;
          }
        }
        return false;
      };
      if (check_ng_shape()) continue;
      ov::element::Type ng_element_type;
      OP_REQUIRES_OK(ctx, util::TFDataTypeToNGraphElementType(
                              tf_input_tensors[i].dtype(), &ng_element_type));

      auto backend = BackendManager::GetBackend();
#if TF_VERSION < 2
      std::shared_ptr<ov::Tensor> ng_tensor =
          make_shared<IETensor>(ng_element_type, ng_shape,
                                (void*)DMAHelper::base(&tf_input_tensors[i]));
#else
      std::shared_ptr<ov::Tensor> ng_tensor = make_shared<IETensor>(
          ng_element_type, ng_shape, tf_input_tensors[i].data());
#endif
      ng_inputs[i] = ng_tensor;
    }
  }

  OVTF_VLOG(4) << "NGraphEncapsulateOp::Compute allocated argument tensors "
                  "for cluster "
               << m_cluster_id;

  // Allocate tensors for the output results.

  auto results = ng_exec->GetResults();
  std::string device;
  Status exec_status = BackendManager::GetBackendName(device);
  if (exec_status != Status::OK()) {
    throw runtime_error(exec_status.error_message());
  }
  auto backend = BackendManager::GetBackend();
  auto dev_type = backend->GetDeviceType();
  std::string precision = dev_type.substr(dev_type.find("_") + 1);
  std::vector<shared_ptr<ov::Tensor>> ng_func_outputs(results.size(), nullptr);
  std::vector<shared_ptr<ov::Tensor>> ng_outputs(results.size(), nullptr);
  std::vector<int> dyn_shape_tensors;
  std::vector<int> output_mappings(results.size(), -1);
  int j = 0;
  if (device != "HDDL") {
    for (auto i = 0; i < results.size(); i++) {
      auto ng_element = results[i];
      if (ng_element->get_output_partial_shape(0).is_dynamic()) {
        OVTF_VLOG(4)
            << "NGraphEncapsulateOp::Compute skipping output allocation for "
               "dynamic tensor at index"
            << i;
        dyn_shape_tensors.push_back(i);
        output_mappings[i] = j;
        j++;
        continue;
      }

      ov::Any any = ng_element->get_rt_info()["index"];
      int64_t output_index = any.as<int64_t>();

      // Create the TF output tensor
      auto ng_shape = ng_element->get_shape();
      TensorShape tf_shape;
      for (auto dim : ng_shape) {
        tf_shape.AddDim(dim);
      }
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(output_index, tf_shape, &output_tensor));

      // Make sure the nGraph-inferred element type agrees with what TensorFlow
      // expected
      ov::element::Type expected_elem_type;
      auto ng_element_type = ng_element->get_element_type();
      if (ng_element_type == ov::element::Type_t::f16 && precision == "FP16")
        ng_element_type = ov::element::Type_t::f32;
      OP_REQUIRES_OK(ctx, util::TFDataTypeToNGraphElementType(
                              ctx->expected_output_dtype(output_index),
                              &expected_elem_type));

      OP_REQUIRES(
          ctx, ng_element_type == expected_elem_type,
          errors::Internal("Element type inferred by nGraph does not match "
                           "the element type expected by TensorFlow"));

#if TF_VERSION < 2
      ng_outputs[i] = make_shared<IETensor>(
          ng_element_type, ng_shape, (void*)DMAHelper::base(output_tensor));

#else
      ng_outputs[i] = make_shared<IETensor>(ng_element_type, ng_shape,
                                            output_tensor->data());
#endif

      auto check_ng_shape = [ng_shape]() {
        if (ng_shape.size() > 0) {
          for (auto dim : ng_shape) {
            if (dim == 0) return true;
          }
        }
        return false;
      };

      if (!(check_ng_shape())) {
        output_mappings[i] = j;
        ng_func_outputs[j++] = ng_outputs[i];
      }
    }
    for (auto i = 0; i < zero_dim_outputs.size(); i++) {
      auto ng_element = zero_dim_outputs[i];
      ov::Any any = ng_element->get_rt_info()["index"];
      int64_t output_index = any.as<int64_t>();

      // Create the TF output tensor
      auto ng_shape = ng_element->get_shape();
      TensorShape tf_shape;
      for (auto dim : ng_shape) {
        tf_shape.AddDim(dim);
      }
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(output_index, tf_shape, &output_tensor));
    }
  }
  OVTF_VLOG(4)
      << "NGraphEncapsulateOp::Compute allocated result tensors for cluster "
      << m_cluster_id;

  int time_create_or_lookup_tensors = create_or_lookup_tensors.ElapsedInMS();
  // Execute the nGraph function.
  int time_execute_function;
  {
    Timer execute_function;
    {
      OVTF_VLOG(4) << "NGraphEncapsulateOp::Compute call starting for cluster "
                   << m_cluster_id;
      try {
        ng_exec->Call(ng_inputs, ng_func_outputs, multi_req_execution);
      } catch (const std::exception& exp) {
        string status_string = "Caught exception while executing cluster " +
                               to_string(m_cluster_id) + ": " +
                               string(exp.what());
        if (NGraphClusterManager::IsClusterFallbackEnabled()) {
          OVTF_VLOG(4) << status_string;
          OP_REQUIRES_OK(ctx, Fallback(ctx));
          return;
        } else {
          OP_REQUIRES(ctx, false, errors::Internal(status_string));
        }
      } catch (...) {
        string status_string = "Caught exception while executing cluster " +
                               to_string(m_cluster_id);
        if (NGraphClusterManager::IsClusterFallbackEnabled()) {
          OVTF_VLOG(4) << status_string;
          OP_REQUIRES_OK(ctx, Fallback(ctx));
          return;
        } else {
          OP_REQUIRES(ctx, false, errors::Internal(status_string));
        }
      }
    }
    time_execute_function = execute_function.ElapsedInMS();
  }

  if (device != "HDDL") {
    for (auto i : dyn_shape_tensors) {
      OP_REQUIRES(ctx, output_mappings[i] != -1,
                  errors::Internal("Mapping error while "
                                   "reading dynamic output blob"));
      auto ng_output = ng_func_outputs[output_mappings[i]];
      // Create the TF output tensor
      auto ng_shape = ng_output->get_shape();
      TensorShape tf_shape;
      for (auto dim : ng_shape) {
        tf_shape.AddDim(dim);
      }

      // TODO: Zero-copy is depreciated temporarily because of memory allocation
      // alignment
      // mismatch related to EIGEN_MAX_ALIGN_BYTES.
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, tf_shape, &output_tensor));

      auto size = ng_output->get_byte_size();
      auto ie_tensor = static_pointer_cast<IETensor>(ng_output);

#if TF_VERSION < 2
      std::copy((uint8_t*)(ng_output->data()),
                ((uint8_t*)(ng_output->data())) + size,
                (uint8_t**)DMAHelper::base(output_tensor));
#else
      std::copy((uint8_t*)(ng_output->data()),
                ((uint8_t*)(ng_output->data())) + size,
                (uint8_t*)(output_tensor->data()));
#endif
    }
  } else {
    for (int i = 0; i < results.size(); i++) {
      auto ng_output = ng_func_outputs[i];

      auto ng_shape = ng_output->get_shape();
      if (results[i]->is_dynamic()) {
        ng_shape = ng_output->get_shape();
      }
      ov::element::Type expected_elem_type;
      auto ng_element = results[i];
      ov::Any any = ng_element->get_rt_info()["index"];
      int64_t output_index = any.as<int64_t>();
      auto ng_element_type = ng_element->get_element_type();
      OP_REQUIRES_OK(ctx, util::TFDataTypeToNGraphElementType(
                              ctx->expected_output_dtype(output_index),
                              &expected_elem_type));
      OP_REQUIRES(
          ctx, ng_element_type == expected_elem_type,
          errors::Internal("Element type inferred by nGraph does not match "
                           "the element type expected by TensorFlow"));
      TensorShape tf_shape;
      for (auto dim : ng_shape) {
        tf_shape.AddDim(dim);
      }

      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(output_index, tf_shape, &output_tensor));

      auto size = ng_output->get_byte_size();
      auto ie_tensor = static_pointer_cast<IETensor>(ng_output);

#if TF_VERSION < 2
      std::copy((uint8_t*)(ng_output->data()),
                ((uint8_t*)(ng_output->data())) + size,
                (uint8_t**)DMAHelper::base(output_tensor));
#else
      std::copy((uint8_t*)(ng_output->data()),
                ((uint8_t*)(ng_output->data())) + size,
                (uint8_t*)(output_tensor->data()));
#endif
    }
    for (auto i = 0; i < zero_dim_outputs.size(); i++) {
      auto ng_element = zero_dim_outputs[i];
      ov::Any any = ng_element->get_rt_info()["index"];
      int64_t output_index = any.as<int64_t>();

      // Create the TF output tensor
      auto ng_shape = ng_element->get_shape();
      TensorShape tf_shape;
      for (auto dim : ng_shape) {
        tf_shape.AddDim(dim);
      }
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(output_index, tf_shape, &output_tensor));
    }
  }

  long vm = 0, rss = 0;
  util::MemoryProfile(vm, rss);
  OVTF_VLOG(1) << "OPENVINO_TF_MEM_PROFILE:  OP_ID: " << m_cluster_id
               << " Step_ID: " << step_id << " Cluster: " << name()
               << " Input Tensors created: "
               << ng_input_tensor_size_in_bytes / (1024 * 1024) << " MB"
               << " Total process memory: " << rss / (1024 * 1024) << " GB";

  OVTF_VLOG(4) << "NGraphEncapsulateOp::Compute call done for cluster "
               << m_cluster_id;

  OVTF_VLOG(4) << "NGraphEncapsulateOp::Compute done marking fresh for cluster "
               << m_cluster_id;
  OVTF_VLOG(1) << "OPENVINO_TF_TIMING_PROFILE: OP_ID: " << m_cluster_id
               << " Step_ID: " << step_id << " Cluster: " << name()
               << " Time-Compute: " << compute_time.ElapsedInMS()
               << " Function-Create-or-Lookup: " << time_func_create_or_lookup
               << " Create-and-copy-tensors: " << time_create_or_lookup_tensors
               << " Execute: " << time_execute_function;
}  // end compute

// Computes signature and gets executable
Status NGraphEncapsulateOp::GetExecutable(
    const std::vector<Tensor>& tf_input_tensors,
    std::shared_ptr<Executable>& ng_exec) {
  auto backend = BackendManager::GetBackend();

  // Compute Signature
  std::vector<const Tensor*> static_input_map;
  std::vector<TensorShape> input_shapes;
  std::stringstream signature_ss;
  for (int i = 0; i < tf_input_tensors.size(); i++) {
    const Tensor& input_tensor = tf_input_tensors[i];
    input_shapes.push_back(input_tensor.shape());
    for (const auto& x : input_tensor.shape()) {
      signature_ss << x.size << ",";
    }
    signature_ss << ";";
  }
  signature_ss << "/";

  static_input_map.resize(tf_input_tensors.size());
  for (int i = 0; i < tf_input_tensors.size(); i++) {
    if (m_input_is_static[i]) {
      static_input_map[i] = &tf_input_tensors[i];
      TF_RETURN_IF_ERROR(
          util::TensorToStream(signature_ss, tf_input_tensors[i]));
      signature_ss << ";";
    }
  }

  string signature = signature_ss.str();
  OVTF_VLOG(5) << "Computed signature: " << signature;
  auto it = m_ng_exec_map.find(signature);
  OVTF_VLOG(4) << "NGraphEncapsulateOp::Compute got inputs for cluster "
               << m_cluster_id;

  // Translate the TensorFlow graph to nGraph.
  std::shared_ptr<ov::Model> ng_function;
  if (it == m_ng_exec_map.end()) {
    // Measure the current total memory usage
    long vm = 0, rss = 0, vm0 = 0, rss0 = 0;
    util::MemoryProfile(vm0, rss0);

    zero_dim_outputs.clear();
    OVTF_VLOG(1) << "Compilation cache miss: " << m_name;
    TF_RETURN_IF_ERROR(Builder::TranslateGraphWithTFFE(
        input_shapes, static_input_map, &m_graph, m_name, ng_function,
        zero_dim_outputs, tf_input_tensors));
    util::DumpNGGraph(ng_function, m_name);

    // Evict the cache if the number of elements exceeds the limit
    std::shared_ptr<Executable> evicted_ng_exec;
    const char* cache_depth_specified =
        std::getenv("OPENVINO_TF_FUNCTION_CACHE_ITEM_DEPTH");
    if (cache_depth_specified != nullptr) {
      m_function_cache_depth_in_items =
          (int)strtol(cache_depth_specified, NULL, 10);
    }
    if (m_ng_exec_map.size() >= m_function_cache_depth_in_items) {
      evicted_ng_exec = m_ng_exec_map[m_lru.back()];
      m_ng_exec_map.erase(m_lru.back());

      m_lru.pop_back();
    }  // cache eviction if cache size greater than cache depth

    try {
      ng_exec = backend->Compile(ng_function);
    } catch (const std::exception& ex) {
      return errors::Internal("Failed to compile function " + m_name + ": ",
                              ex.what());
    }

    m_ng_exec_map[signature] = ng_exec;

    m_lru.push_front(signature);

    // Memory after
    util::MemoryProfile(vm, rss);
    auto delta_vm_mem = vm - vm0;
    auto delta_res_mem = rss - rss0;
    OVTF_VLOG(1) << "OPENVINO_TF_CACHE_PROFILE: OP_ID: " << m_cluster_id
                 << " Cache length: " << m_ng_exec_map.size()
                 << " Cluster: " << m_name << " Delta VM: " << delta_vm_mem
                 << " Delta RSS: " << delta_res_mem
                 << " KB Total RSS: " << rss / (1024 * 1024) << " GB "
                 << " VM: " << vm / (1024 * 1024) << " GB" << endl;
  }  // end of input signature not found in m_ng_exec_map
  else {
    // Found the input signature in m_ng_exec_map, use the cached executable
    // Update the m_lru
    if (signature != m_lru.front()) {
      m_lru.remove(signature);
      m_lru.push_front(signature);
    }
    ng_exec = it->second;
  }
  return Status::OK();
}

Status NGraphEncapsulateOp::Fallback(OpKernelContext* ctx) {
  OVTF_VLOG(1) << "Cluster " << name() << " fallback to native TF runtime ";
  if (!NGraphClusterManager::CheckClusterFallback(m_cluster_id)) {
    NGraphClusterManager::SetClusterFallback(m_cluster_id, true);
    GraphDef* graph_def = NGraphClusterManager::GetClusterGraph(m_cluster_id);
    SessionOptions options;
    std::shared_ptr<tensorflow::Session> session(
        tensorflow::NewSession(options));
    Status session_create_status = session->Create(*graph_def);
    if (!session_create_status.ok()) {
      return session_create_status;
    }
    m_session = session;

    vector<Node*> ordered;
    GetReversePostOrder(m_graph, &ordered, NodeComparatorName());

    vector<const Node*> tf_params;
    vector<const Node*> tf_ret_vals;

    for (const auto n : ordered) {
      if (n->IsSink() || n->IsSource()) {
        continue;
      }

      if (n->IsControlFlow()) {
        return errors::Unimplemented(
            "Encountered a control flow op in the openvino_tensorflow: ",
            n->DebugString());
      }

      if (n->IsArg()) {
        tf_params.push_back(n);
      } else if (n->IsRetval()) {
        tf_ret_vals.push_back(n);
      }
    }
    m_session_input_names.resize(tf_params.size());
    for (auto parm : tf_params) {
      DataType dtype;
      if (GetNodeAttr(parm->attrs(), "T", &dtype) != Status::OK()) {
        return errors::InvalidArgument("No data type defined for _Arg");
      }
      int index;
      if (GetNodeAttr(parm->attrs(), "index", &index) != Status::OK()) {
        return errors::InvalidArgument("No index defined for _Arg");
      }
      m_session_input_names[index] = parm->name();
    }
    m_session_output_names.resize(tf_ret_vals.size());
    for (auto n : tf_ret_vals) {
      if (n->num_inputs() != 1) {
        return errors::InvalidArgument("_Retval has ", n->num_inputs(),
                                       " inputs, should have 1");
      }
      int index;
      if (GetNodeAttr(n->attrs(), "index", &index) != Status::OK()) {
        return errors::InvalidArgument("No index defined for _Retval");
      }
      std::vector<const Edge*> output_edges;
      TF_RETURN_IF_ERROR(n->input_edges(&output_edges));
      m_session_output_names[index] =
          output_edges[0]->src()->name() + ":" +
          std::to_string(output_edges[0]->src_output());
    }
  }

  std::vector<std::pair<string, Tensor>> input_tensor_list(
      m_session_input_names.size());
  for (int i = 0; i < m_session_input_names.size(); i++) {
    input_tensor_list[i] = {m_session_input_names[i], ctx->input(i)};
  }
  std::vector<Tensor> outputs;
  tensorflow::RunOptions run_options;
  run_options.set_inter_op_thread_pool(-1);
  tensorflow::RunMetadata run_metadata;
  Status run_status =
      m_session->Run(run_options, input_tensor_list, m_session_output_names, {},
                     &outputs, &run_metadata);
  if (run_status != Status::OK()) {
    return errors::Internal("Failed to run TF session for " + name());
  }
  for (int i = 0; i < outputs.size(); i++) {
    Tensor* output_tensor = ctx->mutable_output(i);
    if (output_tensor == nullptr) {
      ctx->set_output(i, outputs[i]);
    } else {
#if TF_VERSION < 2
      std::memcpy((void*)(DMAHelper::base(output_tensor)),
                  (void*)(DMAHelper::base(&(outputs[i]))),
                  outputs[i].AllocatedBytes());
#else
      std::memcpy((void*)(output_tensor->data()), (void*)(outputs[i].data()),
                  outputs[i].AllocatedBytes());
#endif
    }
  }
  return Status::OK();
}

}  // namespace openvino_tensorflow

REGISTER_KERNEL_BUILDER(Name("_nGraphEncapsulate").Device(DEVICE_CPU),
                        openvino_tensorflow::NGraphEncapsulateOp);

}  // namespace tensorflow
