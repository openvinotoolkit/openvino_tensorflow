/*******************************************************************************
 * Copyright 2019 Intel Corporation
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
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
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

// Ngraph Encapsulate Implementation class for EncapsulateOp class
//---------------------------------------------------------------------------
//  NGraphEncapsulateImpl::ctor
//---------------------------------------------------------------------------
NGraphEncapsulateImpl::NGraphEncapsulateImpl()
    : m_graph(OpRegistry::Global()), m_freshness_tracker(nullptr) {
  my_instance_id = s_instance_count;
  s_instance_count++;
}

// Use tensorflow input tensors to get input_shapes, static_input_map
// and compute the signature
Status NGraphEncapsulateImpl::ComputeSignature(
    const std::vector<Tensor>& tf_input_tensors,
    std::vector<TensorShape>& input_shapes,
    std::vector<const Tensor*>& static_input_map,
    std::stringstream& signature_ss) {
  // Get the inputs
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
    const Tensor& input_tensor = tf_input_tensors[i];
    if (m_input_is_static[i]) {
      static_input_map[i] = &input_tensor;
      TF_RETURN_IF_ERROR(TensorToStream(signature_ss, input_tensor));
      signature_ss << ";";
    }
  }
  return Status::OK();
}

// Calls ComputeSignature and gets ngraph executable
Status NGraphEncapsulateImpl::GetNgExecutable(
    const std::vector<Tensor>& tf_input_tensors,
    std::vector<TensorShape>& input_shapes,
    std::vector<const Tensor*>& static_input_map,
    ng::runtime::Backend*& op_backend,
    std::shared_ptr<ngraph::runtime::Executable>& ng_exec) {
  std::stringstream signature_ss;
  string signature;

  std::shared_ptr<ngraph::Function> ng_function;
  std::shared_ptr<ngraph::runtime::Executable> evicted_ng_exec;

  NGRAPH_VLOG(4) << "GetNgExecutable: Got backend of type: "
                 << m_op_backend_name;
  op_backend = BackendManager::GetBackend(m_op_backend_name);

  // Compute Signature
  TF_RETURN_IF_ERROR(ComputeSignature(tf_input_tensors, input_shapes,
                                      static_input_map, signature_ss));
  signature = signature_ss.str();

  NGRAPH_VLOG(5) << "Computed signature: " << signature;

  auto it = m_ng_exec_map.find(signature);

  NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute got inputs for cluster "
                 << m_ngraph_cluster;

  // Translate the TensorFlow graph to nGraph.
  if (it == m_ng_exec_map.end()) {
    // Measure the current total memory usage
    long vm, rss, vm0, rss0;
    MemoryProfile(vm0, rss0);

    NGRAPH_VLOG(1) << "Compilation cache miss: " << m_name;
    string serialized_ng_func;
    if (!m_do_aot) {
      TF_RETURN_IF_ERROR(Builder::TranslateGraph(input_shapes, static_input_map,
                                                 &m_graph, ng_function));
      ng_function->set_friendly_name(m_name);
      int json_indentation = 4;
      serialized_ng_func = ngraph::serialize(ng_function, json_indentation);
    } else {
      auto itr = m_aot_functions.find(signature);
      if (itr == m_aot_functions.end()) {
        return errors::Internal(
            "Expected to find AOT precompiled ng function of signature: ",
            signature);
      }
      serialized_ng_func = itr->second;
    }

    // Serialize to nGraph if needed
    if (std::getenv("NGRAPH_ENABLE_SERIALIZE") != nullptr) {
      std::string file_name = "tf_function_" + m_name + ".json";
      TF_RETURN_IF_ERROR(
          StringToFile("tf_function_" + m_name + ".json", serialized_ng_func));
#if defined NGRAPH_DISTRIBUTED
      int rank_id;
      rank_id = ng::get_distributed_interface()->get_rank();
      TF_RETURN_IF_ERROR(StringToFile(
          "tf_function_" + m_name + "_" + to_string(rank_id) + ".json",
          serialized_ng_func));
#endif
    }
    // Evict the cache if the number of elements exceeds the limit
    const char* cache_depth_specified =
        std::getenv("NGRAPH_TF_FUNCTION_CACHE_ITEM_DEPTH");
    if (cache_depth_specified != nullptr) {
      my_function_cache_depth_in_items = atoi(cache_depth_specified);
    }
    if (m_ng_exec_map.size() >= my_function_cache_depth_in_items) {
      int input_tensors_bytes_free = 0;
      evicted_ng_exec = m_ng_exec_map[m_lru.back()];
      m_ng_exec_map.erase(m_lru.back());
      m_serialized_ng_function_map.erase(evicted_ng_exec);

      // Call delete function here for the erased func
      op_backend->remove_compiled_function(evicted_ng_exec);
      // Now clean the input cache
      std::vector<std::pair<void*, std::shared_ptr<ng::runtime::Tensor>>>&
          input_caches = m_ng_exec_input_cache_map[evicted_ng_exec];
      for (auto& next_input : input_caches) {
        input_tensors_bytes_free += next_input.second->get_size_in_bytes();
        next_input.second.reset();
      }
      m_ng_exec_input_cache_map.erase(evicted_ng_exec);

      // Clean the output cache
      std::vector<std::pair<void*, std::shared_ptr<ng::runtime::Tensor>>>&
          output_caches = m_ng_exec_output_cache_map[evicted_ng_exec];
      int output_tensors_bytes_free = 0;
      for (auto& next_output : output_caches) {
        output_tensors_bytes_free += next_output.second->get_size_in_bytes();
        next_output.second.reset();
      }
      m_ng_exec_output_cache_map.erase(evicted_ng_exec);
      m_lru.pop_back();
      NGRAPH_VLOG(1) << "NGRAPH_TF_MEM_PROFILE:  OP_ID: " << my_instance_id
                     << " Cluster: " << m_name << " Input Tensors freed: "
                     << input_tensors_bytes_free / (1024 * 1024) << " MB"
                     << " Output Tensors freed: "
                     << output_tensors_bytes_free / (1024 * 1024) << " MB";
    }  // cache eviction if cache size greater than cache depth

    ngraph::Event event_compile("Compile nGraph", m_name, "");
    BackendManager::LockBackend(m_op_backend_name);
    try {
      if (m_do_aot) {
        auto itr = m_aot_execs.find(signature);
        if (itr == m_aot_execs.end()) {
          BackendManager::UnlockBackend(m_op_backend_name);
          return errors::Internal(
              "Requested AOT, but could not find string with the "
              "signature: ",
              signature);
        }
        stringstream serialized_exec_read;
        serialized_exec_read << (itr->second);
        ng_exec = op_backend->load(serialized_exec_read);
      } else {
        ng_exec = op_backend->compile(ng_function);
      }
    } catch (const std::exception& exp) {
      BackendManager::UnlockBackend(m_op_backend_name);
      Status st = StringToFile("tf_function_error_" + m_name + ".json",
                               serialized_ng_func);
      string status_string =
          "Caught exception while compiling op_backend: " + string(exp.what()) +
          (st.ok() ? "" : (" Also error in dumping serialized function: " +
                           st.error_message()));
      return errors::Internal(status_string);
    } catch (...) {
      BackendManager::UnlockBackend(m_op_backend_name);
      Status st = StringToFile("tf_function_error_" + m_name + ".json",
                               serialized_ng_func);
      string status_string =
          "Error in compiling op_backend." +
          (st.ok() ? "" : (" Also error in dumping serialized function: " +
                           st.error_message()));
      return errors::Internal(status_string);
    }
    BackendManager::UnlockBackend(m_op_backend_name);
    event_compile.Stop();
    ngraph::Event::write_trace(event_compile);

    SetNgExecMap(signature, ng_exec);

    // caching ng_function to serialize to ngraph if needed
    m_serialized_ng_function_map[ng_exec] = serialized_ng_func;

    m_lru.push_front(signature);
    // Memory after
    MemoryProfile(vm, rss);
    auto delta_vm_mem = vm - vm0;
    auto delta_res_mem = rss - rss0;
    NGRAPH_VLOG(1) << "NGRAPH_TF_CACHE_PROFILE: OP_ID: " << my_instance_id
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

// Allocate tensors for input arguments. Creates ngraph input tensors using
// tensorflow tensors required to execute ngraph function
Status NGraphEncapsulateImpl::AllocateNGInputTensors(
    const std::vector<Tensor>& tf_input_tensors,
    const std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
    const PipelinedTensorVector& inp_group_from_pipeline,
    ng::runtime::Backend* const op_backend,
    vector<shared_ptr<ng::runtime::Tensor>>& ng_inputs) {
  std::vector<std::unique_ptr<ngraph::Event>> input_copy_events;
  std::vector<TensorShape> input_shapes;
  std::vector<std::pair<void*, std::shared_ptr<ng::runtime::Tensor>>>&
      input_caches = m_ng_exec_input_cache_map[ng_exec];
  input_caches.resize(tf_input_tensors.size());
#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
  log_copies = false;
  TF_RETURN_IF_ERROR(IsNgraphTFLogTensorCopiesEnabled(m_graph_id, log_copies));
  std::stringstream copy_log_str;
  copy_log_str << "["
               << "NGraphEncapsulate:"
               << "]: " << m_name << " ,GraphID " << m_graph_id << "\n";
  number_of_copies = 0;
#endif

  for (int i = 0; i < tf_input_tensors.size(); i++) {
#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
    bool ref_exists = NGraphCatalog::ExistsInInputVariableSharedNameMap(
        m_graph_id, m_name, i);

    // If the input is from a Variable node, we are dealing with later
    // just add a nullptr to the ng_inputs vector.
    if (ref_exists) {
      NGRAPH_VLOG(4) << "NGraphEncapsulateOp:: Input from Variable Node";
      ng_inputs.push_back(nullptr);
      continue;
    }
    NGRAPH_VLOG(4) << "NGraphEncapsulateOp:: Input from non Variable Node";
#endif
    ng::Shape ng_shape(tf_input_tensors[i].shape().dims());
    for (int j = 0; j < tf_input_tensors[i].shape().dims(); ++j) {
      ng_shape[j] = tf_input_tensors[i].shape().dim_size(j);
    }
    ng::element::Type ng_element_type;
    TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(
        tf_input_tensors[i].dtype(), &ng_element_type));

    // At the first call of the ng_exec, both last_src_ptr and
    // last_ng_tensor shall point to null. Otherwise, they are retrived
    // from cache.
    void* last_src_ptr = input_caches[i].first;
    std::shared_ptr<ng::runtime::Tensor> last_ng_tensor =
        input_caches[i].second;
    void* current_src_ptr = (void*)DMAHelper::base(&tf_input_tensors[i]);
    std::shared_ptr<ng::runtime::Tensor> current_ng_tensor = GetCurrentNgTensor(
        current_src_ptr, last_src_ptr, last_ng_tensor, false, ng_exec,
        op_backend, ng_element_type, ng_shape,
        m_executable_can_create_tensor ? inp_group_from_pipeline[i] : nullptr);
    bool is_cpu = m_op_backend_name == "CPU";

    if (!is_cpu && current_ng_tensor->get_stale()) {
      // Fresh or stale, in case of CPU this step is never needed
      try {
#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
        int copies = number_of_copies;
        SetNumberOfCopies(copies++);
        copy_log_str << " COPY_INP_VAL[" << i << "]";
#endif
        size_t copy_size =
            current_ng_tensor->get_element_count() * ng_element_type.size();
        string event_name =
            "Input_" + to_string(i) + "_" + to_string(copy_size);
        std::unique_ptr<ngraph::Event> event_copy_input_next(
            new ngraph::Event(event_name, m_name, ""));
        current_ng_tensor->write(
            current_src_ptr, 0,
            current_ng_tensor->get_element_count() * ng_element_type.size());

        event_copy_input_next->Stop();
        input_copy_events.push_back(std::move(event_copy_input_next));

      } catch (const std::exception& exp) {
        return errors::Internal(
            "Caught exception while transferring tensor data to nGraph. "
            "Exception: ",
            exp.what());
      } catch (...) {
        return errors::Internal(
            "Error in transferring tensor data to nGraph\n");
      }
    }
    input_caches[i] = std::make_pair(current_src_ptr, current_ng_tensor);
    ng_inputs.push_back(current_ng_tensor);
  }  // for (int i = 0; i < input_shapes.size(); i++)

  // Now write the events back
  for (auto& next : input_copy_events) {
    ngraph::Event::write_trace(*next.get());
  }
  return Status::OK();
}

// Allocate tensors for output results.  Creates ngraph output tensors using
// tensorflow tensors required to execute ngraph function
Status NGraphEncapsulateImpl::AllocateNGOutputTensors(
    const std::vector<Tensor*>& output_tensors,
    const std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
    const PipelinedTensorVector& out_group_from_pipeline,
    ng::runtime::Backend* const op_backend,
    vector<shared_ptr<ng::runtime::Tensor>>& ng_outputs) {
  std::vector<std::pair<void*, std::shared_ptr<ng::runtime::Tensor>>>&
      output_caches = m_ng_exec_output_cache_map[ng_exec];
  output_caches.resize(ng_exec->get_results().size());

  // ngraph executable returns get_results, using that to get the tensor shape
  // and element type.
  for (auto i = 0; i < ng_exec->get_results().size(); i++) {
    auto ng_element = ng_exec->get_results()[i];
    auto ng_shape = ng_element->get_shape();
    auto ng_element_type = ng_element->get_element_type();

    void* last_dst_ptr = output_caches[i].first;
    std::shared_ptr<ng::runtime::Tensor> last_ng_tensor =
        output_caches[i].second;

    void* current_dst_ptr = DMAHelper::base(output_tensors[i]);
    std::shared_ptr<ng::runtime::Tensor> current_ng_tensor = nullptr;

#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
    bool ref_exists =
        NGraphCatalog::ExistsInEncapOutputInfoMap(m_graph_id, m_name, i);

    // if the output tensor is going to be assigned to a variable
    // we are dealing with later, just add a nullptr to ng_outputs vectorç
    if (ref_exists) {
      NGRAPH_VLOG(4) << "NGraphEncapsulateImpl:: Output from Variable Node";
      ng_outputs.push_back(nullptr);
      continue;
    }
    NGRAPH_VLOG(4) << "NGraphEncapsulateImpl:: Output from non Variable Node";
#endif

    current_ng_tensor = GetCurrentNgTensor(
        current_dst_ptr, last_dst_ptr, last_ng_tensor, true, ng_exec,
        op_backend, ng_element_type, ng_shape,
        m_executable_can_create_tensor ? out_group_from_pipeline[i] : nullptr);

    current_ng_tensor->set_stale(true);
    output_caches[i] = std::make_pair(current_dst_ptr, current_ng_tensor);
    ng_outputs.push_back(current_ng_tensor);
  }

  return Status::OK();
}

// Get current ngraph tensor
std::shared_ptr<ng::runtime::Tensor> NGraphEncapsulateImpl::GetCurrentNgTensor(
    void* current_tf_ptr, void* last_tf_ptr,
    const std::shared_ptr<ng::runtime::Tensor>& last_ng_tensor,
    const bool& output_tensor,
    const std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
    ng::runtime::Backend* const op_backend,
    const ng::element::Type& ng_element_type, const ng::Shape& ng_shape,
    std::shared_ptr<ng::runtime::Tensor> tensor_from_pipeline) {
  // NOTE: we assume that TF's pointers WILL change if it actually changes
  // values. ie, it will not reuse the same space if its rewritten it
  bool tf_tensor_has_changed = current_tf_ptr != last_tf_ptr;
  bool no_ng_tensor_found = last_ng_tensor == nullptr;
  bool is_cpu = m_op_backend_name == "CPU";

  // We need to check last_ng_tensor != nullptr, since there are cases where
  // at the first call to the ng_exec, both current_dst_ptr (when the
  // output is a 0-sized tensor) and last_dst_ptr (uninitialized at the
  // first call) are nullptr
  // A new tensor needs to be created for sure if no_ng_tensor_found
  // Additionally for CPU, it needs to be created if tf_tensor_has_changed,
  // for others, we do not create
  bool need_new_tensor_creation;
  if (is_cpu) {
    need_new_tensor_creation = no_ng_tensor_found || tf_tensor_has_changed;
  } else {
    need_new_tensor_creation = no_ng_tensor_found;
  }

  // It is stale if a new tensor was created OR the tf tensor has changed OR
  // (tf tensor has not changed, but freshness tracker says its stale)
  bool is_stale;
  if (output_tensor) {
    is_stale = true;  // For output tensors, it is always set stale to true
  } else {
    is_stale = need_new_tensor_creation || tf_tensor_has_changed ||
               (!tf_tensor_has_changed &&
                !m_freshness_tracker->IsFresh(current_tf_ptr, ng_exec));
  }
  // create a new ng tensor or use the last one
  std::shared_ptr<ng::runtime::Tensor> current_ng_tensor;
  if (m_executable_can_create_tensor) {
    current_ng_tensor = tensor_from_pipeline;
  } else {
    if (need_new_tensor_creation) {
      if (is_cpu) {
        current_ng_tensor = op_backend->create_tensor(ng_element_type, ng_shape,
                                                      current_tf_ptr);
      } else {
        current_ng_tensor =
            op_backend->create_tensor(ng_element_type, ng_shape);
      }
    } else {
      current_ng_tensor = last_ng_tensor;
    }
  }
  current_ng_tensor->set_stale(is_stale);
  return current_ng_tensor;
}

Status NGraphEncapsulateImpl::ParseNodeAttributes(
    const google::protobuf::Map<string, AttrValue>& additional_attributes,
    std::unordered_map<std::string, std::string>* additional_attribute_map) {
  for (auto itx : additional_attributes) {
    // Find the optional attributes to be sent to the backend.
    // The optional attributes have '_ngraph_' appended to the start
    // so we need to get rid of that and only send the remaining string
    // since the backend will only look for that.
    // '_ngraph_' is only appended for the bridge.
    // For e.g. _ngraph_ice_cores --> ice_cores
    if (itx.first.find("_ngraph_") != std::string::npos) {
      // TODO: decide what the node attributes should be.
      // right now _ngraph_aot_ is used by aot, _ngraph_ is used for optional
      // attributes
      auto attr_name = itx.first;
      auto attr_value = itx.second.s();
      if (attr_name.find("_ngraph_aot_") != std::string::npos) {
        // The string is in the format: _ngraph_aot_ngexec_signature or
        // _ngraph_aot_ngfunction_signature or _ngraph_aot_requested
        // TODO: do not pass these 3 attributes to set_config of backend
        if (attr_name.find("_ngraph_aot_ngexec_") != std::string::npos) {
          m_aot_execs[ng::split(attr_name, '_')[4]] = attr_value;
        } else if (attr_name.find("_ngraph_aot_ngfunction_") !=
                   std::string::npos) {
          // The other option is _ngraph_aot_ngfunction_
          // No need to save or do anything with _ngraph_aot_ngfunction_. They
          // are there for debugging only
          m_aot_functions[ng::split(attr_name, '_')[4]] = attr_value;
        } else if (attr_name.find("_ngraph_aot_requested") !=
                   std::string::npos) {
          m_do_aot = (attr_value == "1");
          if (m_do_aot) {
            NGRAPH_VLOG(1) << "Using AOT for encapsulate " +
                                  to_string(m_ngraph_cluster);
          }
        } else {
          return errors::Internal(
              "Ngraph attribues beginning with _ngraph_aot_ "
              "must be _ngraph_aot_ngexec_<signature> or "
              "_ngraph_aot_ngfunction_<signature>. But got "
              "attribute named: ",
              itx.first);
        }
      } else {
        NGRAPH_VLOG(4) << "Attribute: " << attr_name.substr(strlen("_ngraph_"))
                       << " Value: " << attr_value;
        additional_attribute_map->insert(
            {attr_name.substr(strlen("_ngraph_")), attr_value});
      }
    }
  }
  if (((m_aot_functions.size() > 0) || (m_aot_execs.size() > 0)) && !m_do_aot) {
    return errors::Internal("The encapsulate ", m_name,
                            " has ngraph functions or executables embedded "
                            "in it, even though AOT was not requested.");
  }
  return Status::OK();
}

Status NGraphEncapsulateImpl::UpdatePipelinedTensorCache(
    std::shared_ptr<ngraph::runtime::Executable> ng_exec) {
  if (!m_executable_can_create_tensor) {
    return errors::Internal(
        "UpdatePipelinedTensorCache called, but executable cannot create "
        "tensors");
  }
  auto itr = m_executable_pipelined_tensors_map.find(ng_exec);
  if (itr == m_executable_pipelined_tensors_map.end()) {
    // Create these pipelined ng tensors only if needed, else reuse from cache
    size_t num_inputs = ng_exec->get_parameters().size();
    size_t num_outputs = ng_exec->get_results().size();
    PipelinedTensorMatrix pipelined_input_tensors(num_inputs);
    PipelinedTensorMatrix pipelined_output_tensors(num_outputs);
    for (size_t i = 0; i < num_inputs; i++) {
      pipelined_input_tensors[i] = ng_exec->create_input_tensor(i, m_depth);
    }
    for (size_t i = 0; i < num_outputs; i++) {
      pipelined_output_tensors[i] = ng_exec->create_output_tensor(i, m_depth);
    }
    m_executable_pipelined_tensors_map.insert(
        {ng_exec, PipelinedTensorsStore(pipelined_input_tensors,
                                        pipelined_output_tensors)});
  }
  return Status::OK();
}

std::tuple<int, PipelinedTensorVector, PipelinedTensorVector>
NGraphEncapsulateImpl::GetTensorsFromPipeline(
    std::shared_ptr<ngraph::runtime::Executable> ng_exec) {
  PipelinedTensorsStore pts = m_executable_pipelined_tensors_map.at(ng_exec);

  // TODO: do something about this spin lock
  // get_tensors returns an index integer, that can be -1, 0, ... depth-1
  // If it returns -1, then it indicates there are no free groups of tensors
  // or the pipeline is full. In that case, we need to wait, hence the while
  std::tuple<int, PipelinedTensorVector, PipelinedTensorVector> out_tpl;
  while (true) {
    out_tpl = pts.get_tensors();

    if (std::get<0>(out_tpl) >= 0) {
      break;
    }
  }
  return out_tpl;
}

Status NGraphEncapsulateImpl::DumpNgFunction(
    const string& file_name,
    std::shared_ptr<ngraph::runtime::Executable> ng_exec) {
  auto itr = m_serialized_ng_function_map.find(ng_exec);
  if (itr == m_serialized_ng_function_map.end()) {
    return errors::Internal(
        "Did not find requested executable in map for exec->serialized ngraph "
        "function when dumping ngraph function");
  }
  return StringToFile(file_name, itr->second);
}

void NGraphEncapsulateImpl::NGraphEncapsulateImpl::ClearExecMaps() {
  m_ng_exec_input_cache_map.clear();
  m_ng_exec_output_cache_map.clear();
  m_ng_exec_map.clear();
  m_serialized_ng_function_map.clear();
  m_executable_pipelined_tensors_map.clear();
}

Status NGraphEncapsulateImpl::GetPipelineIdxAndTensors(
    const std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
    std::tuple<int, PipelinedTensorVector, PipelinedTensorVector>& tpl) {
  // Check if the pipelined tensors for this executable is created
  // If not create and add them to the cache
  TF_RETURN_IF_ERROR(UpdatePipelinedTensorCache(ng_exec));
  // Cache must contain the ng_exec at this point

  try {
    tpl = GetTensorsFromPipeline(ng_exec);
  } catch (const std::exception& exp) {
    return errors::Internal(
        "Caught exception while getting pipelined tensors: ", exp.what(), "\n");
  }

  auto pipeline_idx = get<0>(tpl);
  if (pipeline_idx < 0) {
    return errors::Internal(
        "Expected GetTensorsFromPipeline to return an index >= 0, but got ",
        pipeline_idx);
  }
  return Status::OK();
}

Status NGraphEncapsulateImpl::ReturnPipelinedTensors(
    std::shared_ptr<ngraph::runtime::Executable> ng_exec, size_t idx) {
  try {
    m_executable_pipelined_tensors_map.at(ng_exec).return_tensors(idx);
  } catch (const std::exception& exp) {
    return errors::Internal(
        "Caught exception while returning pipelined tensors: ", exp.what(),
        "\n");
  }
  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow