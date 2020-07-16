/*******************************************************************************
 * Copyright 2019-2020 Intel Corporation
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

#include "logging/ngraph_log.h"
#include "ngraph_bridge/ngraph_backend.h"
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

// Ngraph Encapsulate Implementation class for EncapsulateOp class
//---------------------------------------------------------------------------
//  NGraphEncapsulateImpl::ctor
//---------------------------------------------------------------------------
NGraphEncapsulateImpl::NGraphEncapsulateImpl() : m_graph(OpRegistry::Global()) {
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

// Compiles the ngraph function and returns ngraph executable
Status NGraphEncapsulateImpl::Compile(
    const std::string& backend_name,
    std::shared_ptr<ngraph::Function> ng_function,
    std::shared_ptr<Executable>& ng_exec) {
  Backend* op_backend = BackendManager::GetBackend(backend_name);
  BackendManager::LockBackend(backend_name);
  try {
    ng_exec = op_backend->compile(ng_function);
  } catch (const std::exception& ex) {
    BackendManager::UnlockBackend(backend_name);
    string fn_name = ng_function->get_friendly_name();
    NgraphSerialize("tf_function_" + fn_name + ".json", ng_function);
    BackendManager::ReleaseBackend(backend_name);
    return errors::Internal("Failed to compile ng_function: ", ex.what());
  }
  BackendManager::UnlockBackend(backend_name);
  return Status::OK();
}

// Compiles the ngraph function and returns ngraph executable as a string
Status NGraphEncapsulateImpl::GetCompiledString(
    const std::string& backend_name,
    std::shared_ptr<ngraph::Function> ng_function, std::string* ng_exec_str) {
  std::shared_ptr<Executable> ng_exec;
  NGraphEncapsulateImpl::Compile(backend_name, ng_function, ng_exec);
  stringstream exec_dump;
  ng_exec->save(exec_dump);
  *ng_exec_str = exec_dump.str();
  return Status::OK();
}

// Calls ComputeSignature and gets ngraph executable
Status NGraphEncapsulateImpl::GetNgExecutable(
    const std::vector<Tensor>& tf_input_tensors,
    std::vector<TensorShape>& input_shapes,
    std::vector<const Tensor*>& static_input_map,
    std::shared_ptr<Executable>& ng_exec) {
  std::stringstream signature_ss;
  string signature;

  std::shared_ptr<ngraph::Function> ng_function;
  std::shared_ptr<Executable> evicted_ng_exec;

  NGRAPH_VLOG(4) << "GetNgExecutable: Got backend of type: "
                 << m_op_backend_name;
  Backend* op_backend = BackendManager::GetBackend(m_op_backend_name);

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
    }
    // Evict the cache if the number of elements exceeds the limit
    const char* cache_depth_specified =
        std::getenv("NGRAPH_TF_FUNCTION_CACHE_ITEM_DEPTH");
    if (cache_depth_specified != nullptr) {
      my_function_cache_depth_in_items = atoi(cache_depth_specified);
    }
    if (m_ng_exec_map.size() >= my_function_cache_depth_in_items) {
      evicted_ng_exec = m_ng_exec_map[m_lru.back()];
      m_ng_exec_map.erase(m_lru.back());
      m_serialized_ng_function_map.erase(evicted_ng_exec);

      // Call delete function here for the erased func
      op_backend->remove_compiled_function(evicted_ng_exec);

      m_lru.pop_back();
    }  // cache eviction if cache size greater than cache depth

    NG_TRACE("Compile nGraph", m_name, "");
    if (m_do_aot) {
      auto itr = m_aot_execs.find(signature);
      if (itr == m_aot_execs.end()) {
        return errors::Internal(
            "Requested AOT, but could not find string with the "
            "signature: ",
            signature);
      }

      BackendManager::LockBackend(m_op_backend_name);
      try {
        stringstream serialized_exec_read;
        serialized_exec_read << (itr->second);
        ng_exec = op_backend->load(serialized_exec_read);
      } catch (const std::exception& exp) {
        BackendManager::UnlockBackend(m_op_backend_name);
        Status st = StringToFile("tf_function_error_" + m_name + ".json",
                                 serialized_ng_func);
        string status_string =
            "Caught exception while compiling op_backend: " +
            string(exp.what()) +
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
    } else {
      TF_RETURN_IF_ERROR(NGraphEncapsulateImpl::Compile(m_op_backend_name,
                                                        ng_function, ng_exec));
    }

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
    const std::shared_ptr<Executable>& ng_exec,
    vector<shared_ptr<ng::runtime::Tensor>>& ng_inputs) {
  Backend* op_backend = BackendManager::GetBackend(m_op_backend_name);
  for (int i = 0; i < tf_input_tensors.size(); i++) {
    ng::Shape ng_shape(tf_input_tensors[i].shape().dims());
    for (int j = 0; j < tf_input_tensors[i].shape().dims(); ++j) {
      ng_shape[j] = tf_input_tensors[i].shape().dim_size(j);
    }
    ng::element::Type ng_element_type;
    TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(
        tf_input_tensors[i].dtype(), &ng_element_type));

    void* current_tf_ptr = (void*)DMAHelper::base(&tf_input_tensors[i]);
    std::shared_ptr<ng::runtime::Tensor> current_ng_tensor =
        op_backend->create_tensor(ng_element_type, ng_shape, current_tf_ptr);
    ng_inputs.push_back(current_ng_tensor);
  }
  return Status::OK();
}

// Allocate tensors for output results.  Creates ngraph output tensors using
// tensorflow tensors required to execute ngraph function
Status NGraphEncapsulateImpl::AllocateNGOutputTensors(
    const std::vector<Tensor*>& output_tensors,
    const std::shared_ptr<Executable>& ng_exec,
    vector<shared_ptr<ng::runtime::Tensor>>& ng_outputs) {
  Backend* op_backend = BackendManager::GetBackend(m_op_backend_name);
  for (auto i = 0; i < ng_exec->get_results().size(); i++) {
    auto ng_element = ng_exec->get_results()[i];
    auto ng_shape = ng_element->get_shape();
    auto ng_element_type = ng_element->get_element_type();

    void* current_tf_ptr = DMAHelper::base(output_tensors[i]);
    std::shared_ptr<ng::runtime::Tensor> current_ng_tensor =
        op_backend->create_tensor(ng_element_type, ng_shape, current_tf_ptr);
    ng_outputs.push_back(current_ng_tensor);
  }
  return Status::OK();
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

Status NGraphEncapsulateImpl::DumpNgFunction(
    const string& file_name, std::shared_ptr<Executable> ng_exec) {
  auto itr = m_serialized_ng_function_map.find(ng_exec);
  if (itr == m_serialized_ng_function_map.end()) {
    return errors::Internal(
        "Did not find requested executable in map for exec->serialized ngraph "
        "function when dumping ngraph function");
  }
  return StringToFile(file_name, itr->second);
}

void NGraphEncapsulateImpl::NGraphEncapsulateImpl::ClearExecMaps() {
  m_ng_exec_map.clear();
  m_serialized_ng_function_map.clear();
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
