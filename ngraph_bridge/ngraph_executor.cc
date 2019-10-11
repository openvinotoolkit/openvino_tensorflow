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
#include "ngraph_bridge/ngraph_executor.h"
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

//---------------------------------------------------------------------------
//  NGraphExecutor::ctor
//---------------------------------------------------------------------------
NGraphExecutor::NGraphExecutor(int instance_id, int cluster_id, int graph_id,
                               unique_ptr<tensorflow::Graph>& graph,
                               const string& backend_name)
    : m_instance_id(instance_id),
      m_ngraph_cluster_id(cluster_id),
      m_graph_id(graph_id),
      m_graph(std::move(graph)),
      m_op_backend_name(backend_name) {
  // Sanity checks
  if (m_graph == nullptr) {
    throw std::runtime_error("Graph is nullptr!");
  }
  NGRAPH_VLOG(3) << "NGraphExecutor(): " << instance_id
                 << " Backend: " << backend_name;

  // Get the backend. Note that the backend may not be available
  // so that's a programming error.
  try {
    auto backend = BackendManager::GetBackend(m_op_backend_name);
    m_executable_can_create_tensor = backend->executable_can_create_tensors();
  } catch (...) {
    throw std::runtime_error(string("Requested backend: '") +
                             m_op_backend_name + string("' not available."));
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

  for (auto node : m_graph->nodes()) {
    if (node->type_string() == "_Arg") {
      arg_nodes.push_back(node);

      int32 index;
      auto status = GetNodeAttr(node->attrs(), "index", &index);
      if (status != Status::OK()) {
        throw std::runtime_error("error getting node attribute index");
      }

      if (index > max_arg_index) {
        max_arg_index = index;
      }
    }
  }

  int size = max_arg_index + 1;
  m_input_is_static.resize(size);

  for (int i = 0; i < size; i++) {
    m_input_is_static[i] = false;
  }

  // Fill the vector.
  for (auto node : arg_nodes) {
    int32 index;
    auto status = GetNodeAttr(node->attrs(), "index", &index);
    if (status != Status::OK()) {
      throw std::runtime_error("error getting node attribute index");
    }

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
}

//---------------------------------------------------------------------------
//  NGraphExecutor::~NGraphExecutor
//---------------------------------------------------------------------------
NGraphExecutor::~NGraphExecutor() {
  auto backend = BackendManager::GetBackend(m_op_backend_name);
  for (auto& next : m_ng_exec_map) {
    // First remove the pipelined tensors
    auto ng_exec = next.second;
    auto tensor_store = m_executable_pipelined_tensors_map.find(ng_exec);
    if (tensor_store == m_executable_pipelined_tensors_map.end()) {
      // There should have been an entry in this Map
      NGRAPH_VLOG(0)
          << "The Pipelied Tensor map is empty for the current executor";
      continue;
    }

    auto io_tensor_tuple = tensor_store->second->get_tensors();
    while (std::get<0>(io_tensor_tuple) > 0) {
      // At this stage everyone must have returned the tensors back to store.
      // So the id must be non negative
      for (auto& input_tensor : std::get<1>(io_tensor_tuple)) {
        input_tensor.reset();
      }
      for (auto& output_tensor : std::get<2>(io_tensor_tuple)) {
        output_tensor.reset();
      }

      io_tensor_tuple = tensor_store->second->get_tensors();
    }

    // Now remove the entry from the function cache
    backend->remove_compiled_function(ng_exec);

    // Finally reset the shared_ptr so that this is deleted by the
    // backend when needed
    ng_exec.reset();
  }

  m_executable_pipelined_tensors_map.clear();
  m_ng_function_map.clear();
  m_ng_exec_map.clear();
}

//---------------------------------------------------------------------------
//  NGraphExecutor::ComputeSignature
//---------------------------------------------------------------------------
Status NGraphExecutor::ComputeSignature(
    const std::vector<Tensor>& tf_input_tensors,
    std::vector<TensorShape>& input_shapes,
    std::vector<const Tensor*>& static_input_map,
    std::stringstream& signature_ss) const {
  // Use tensorflow input tensors to get input_shapes, static_input_map
  // and compute the signature
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

//---------------------------------------------------------------------------
//  NGraphExecutor::GetNgExecutable
//---------------------------------------------------------------------------
Status NGraphExecutor::GetNgExecutable(
    const std::vector<Tensor>& tf_input_tensors,
    std::shared_ptr<ngraph::runtime::Executable>& ng_exec, bool& cache_hit) {
  NGRAPH_VLOG(4) << "GetNgExecutable: Got backend of type: "
                 << m_op_backend_name;

  // Get the backend. Note that the backend may not be available
  // so that's a programmng error.
  ng::runtime::Backend* op_backend;
  try {
    op_backend = BackendManager::GetBackend(m_op_backend_name);
  } catch (...) {
    return errors::Internal("Backend not available: ", m_op_backend_name);
  }

  lock_guard<mutex> lock(m_mutex);

  std::stringstream signature_ss;
  std::vector<TensorShape> input_shapes;
  std::vector<const Tensor*> static_input_map;
  TF_RETURN_IF_ERROR(ComputeSignature(tf_input_tensors, input_shapes,
                                      static_input_map, signature_ss));
  string signature;
  signature = signature_ss.str();

  NGRAPH_VLOG(5) << "Computed signature: " << signature;

  cache_hit = false;
  auto it = m_ng_exec_map.find(signature);

  // Translate the TensorFlow graph to nGraph.
  if (it == m_ng_exec_map.end()) {
    // Measure the current total memory usage
    long vm, rss, vm0, rss0;
    MemoryProfile(vm0, rss0);

    NGRAPH_VLOG(1) << "Compilation cache miss: " << m_node_name;

    std::shared_ptr<ngraph::Function> ng_function;
    std::shared_ptr<ngraph::runtime::Executable> evicted_ng_exec;

    if (!m_do_aot) {
      TF_RETURN_IF_ERROR(Builder::TranslateGraph(input_shapes, static_input_map,
                                                 m_graph.get(), ng_function));
      ng_function->set_friendly_name(m_node_name);
    } else {
      auto itr = m_aot_functions.find(signature);
      if (itr == m_aot_functions.end()) {
        return errors::Internal(
            "Expected to find AOT precompiled ng function of signature: ",
            signature);
      }
      ng_function = ng::deserialize(itr->second);
    }

    auto function_size = ng_function->get_graph_size() / 1024;  // kb unit

    // Serialize to nGraph if needed
    if (std::getenv("NGRAPH_ENABLE_SERIALIZE") != nullptr) {
      std::string file_name = "tf_function_" + m_node_name + ".json";
      NgraphSerialize("tf_function_" + m_node_name + ".json", ng_function);
#if defined NGRAPH_DISTRIBUTED
      int rank_id;
      rank_id = ng::get_distributed_interface()->get_rank();
      NgraphSerialize(
          "tf_function_" + m_node_name + "_" + to_string(rank_id) + ".json",
          ng_function);
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
      m_ng_function_map.erase(evicted_ng_exec);

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
      NGRAPH_VLOG(1) << "NGRAPH_TF_MEM_PROFILE:  OP_ID: " << m_instance_id
                     << " Cluster: " << m_node_name << " Input Tensors freed: "
                     << input_tensors_bytes_free / (1024 * 1024) << " MB"
                     << " Output Tensors freed: "
                     << output_tensors_bytes_free / (1024 * 1024) << " MB";
    }  // cache eviction if cache size greater than cache depth

    ngraph::Event event_compile("Compile nGraph", m_node_name, "");
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
      NgraphSerialize("tf_function_error_" + m_node_name + ".json",
                      ng_function);
      return errors::Internal("Caught exception while compiling op_backend: ",
                              exp.what(), "\n");
    } catch (...) {
      BackendManager::UnlockBackend(m_op_backend_name);
      NgraphSerialize("tf_function_error_" + m_node_name + ".json",
                      ng_function);
      return errors::Internal("Error in compiling op_backend\n");
    }
    BackendManager::UnlockBackend(m_op_backend_name);
    event_compile.Stop();
    ngraph::Event::write_trace(event_compile);
    m_ng_exec_map[signature] = ng_exec;

    // caching ng_function to serialize to ngraph if needed
    m_ng_function_map[ng_exec] = ng_function;

    m_lru.push_front(signature);
    // Memory after
    MemoryProfile(vm, rss);
    auto delta_vm_mem = vm - vm0;
    auto delta_res_mem = rss - rss0;
    NGRAPH_VLOG(1) << "NGRAPH_TF_CACHE_PROFILE: OP_ID: " << m_instance_id
                   << " Cache length: " << m_ng_exec_map.size()
                   << "  Cluster: " << m_node_name
                   << " Delta VM: " << delta_vm_mem
                   << "  Delta RSS: " << delta_res_mem
                   << "  Function size: " << function_size
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
    cache_hit = true;
    NGRAPH_VLOG(1) << "Compilation cache hit: " << m_node_name;
  }
  return Status::OK();
}

//---------------------------------------------------------------------------
//  GetNgFunction
//---------------------------------------------------------------------------
Status NGraphExecutor::GetNgFunction(
    const std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
    std::shared_ptr<ngraph::Function>& ng_function) {
  // Lookup the function from the Map
  auto it = m_ng_function_map.find(ng_exec);
  if (it == m_ng_function_map.end()) {
    errors::Internal("Function not found for this executable");
  }
  ng_function = it->second;
  return Status::OK();
}

//---------------------------------------------------------------------------
//  ParseNodeAttributes
//---------------------------------------------------------------------------
Status NGraphExecutor::ParseNodeAttributes(
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
                                  to_string(m_ngraph_cluster_id);
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
    return errors::Internal("The encapsulate ", m_node_name,
                            " has ngraph functions or executables embedded "
                            "in it, even though AOT was not requested.");
  }
  return Status::OK();
}

//---------------------------------------------------------------------------
//  InitializeIOTensorPipeline
//---------------------------------------------------------------------------
Status NGraphExecutor::InitializeIOTensorPipeline(
    std::shared_ptr<ngraph::runtime::Executable> ng_exec) {
  if (!m_executable_can_create_tensor) {
    return errors::Internal(
        "InitializeIOTensorPipeline called, but executable cannot create "
        "tensors");
  }

  lock_guard<mutex> lock(m_mutex);
  auto itr = m_executable_pipelined_tensors_map.find(ng_exec);
  if (itr == m_executable_pipelined_tensors_map.end()) {
    // Create these pipelined ng tensors only if needed, else reuse from cache
    size_t num_inputs = ng_exec->get_parameters().size();
    size_t num_outputs = ng_exec->get_results().size();

    if (num_inputs == 0 || num_outputs == 0) {
      return errors::Internal("Bad input/output length. Input size: ",
                              num_inputs, " Output size: ", num_outputs);
    }

    // If the input or the output size if 0 then???
    NGRAPH_VLOG(5) << "InitializeIOTensorPipeline: In: " << num_inputs
                   << " Out: " << num_outputs;
    PipelinedTensorMatrix pipelined_input_tensors(num_inputs);
    PipelinedTensorMatrix pipelined_output_tensors(num_outputs);
    for (size_t i = 0; i < num_inputs; i++) {
      pipelined_input_tensors[i] = ng_exec->create_input_tensor(i, m_depth);
    }
    for (size_t i = 0; i < num_outputs; i++) {
      pipelined_output_tensors[i] = ng_exec->create_output_tensor(i, m_depth);
    }
    shared_ptr<PipelinedTensorsStore> pts(new PipelinedTensorsStore(
        pipelined_input_tensors, pipelined_output_tensors));
    m_executable_pipelined_tensors_map.insert({ng_exec, pts});
  }
  return Status::OK();
}

//---------------------------------------------------------------------------
//  GetTensorsFromPipeline
//---------------------------------------------------------------------------
Status NGraphExecutor::GetTensorsFromPipeline(
    const std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
    std::tuple<int, PipelinedTensorVector, PipelinedTensorVector>& io_tensors) {
  auto status = InitializeIOTensorPipeline(ng_exec);
  if (status != Status::OK()) {
    return status;
  }

  // Lookup the executable
  lock_guard<mutex> lock(m_mutex);
  PipelinedTensorsStore* pts(nullptr);
  try {
    const auto& item = m_executable_pipelined_tensors_map.at(ng_exec);
    pts = item.get();
  } catch (...) {
    return errors::Internal("Executable not found in the cache");
  }

  // get_tensors returns an index integer, that can be -1, 0, ... depth-1
  // If it returns -1, then it indicates there are no free groups of tensors
  // or the pipeline is full.
  io_tensors = pts->get_tensors();
  if (std::get<0>(io_tensors) < 0) {
    return errors::Internal("No free tensor available");
  }

  return Status::OK();
}
}  // namespace ngraph_bridge
}  // namespace tensorflow