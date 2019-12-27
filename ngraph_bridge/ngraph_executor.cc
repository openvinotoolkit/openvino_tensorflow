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
#include "ngraph_bridge/ngraph_data_cache.h"
#include "ngraph_bridge/ngraph_executor.h"
#include "ngraph_bridge/ngraph_mark_for_clustering.h"
#include "ngraph_bridge/ngraph_timer.h"
#include "ngraph_bridge/ngraph_utils.h"
#include "ngraph_bridge/ngraph_var.h"

#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
#include "ngraph_bridge/ngraph_catalog.h"
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
                               const string& backend_name,
                               const string& node_name, const int cache_depth)
    : m_instance_id(instance_id),
      m_ngraph_cluster_id(cluster_id),
      m_graph_id(graph_id),
      m_graph(std::move(graph)),
      m_op_backend_name(backend_name),
      m_node_name(node_name),
      m_ng_data_cache(cache_depth) {
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

  // Some error checking before refactoring the above code
  int number_of_inputs = FindNumberOfNodes(m_graph.get(), "_Arg");
  int number_of_outputs = FindNumberOfNodes(m_graph.get(), "_Retval");

  if (number_of_inputs != size) {
    throw std::runtime_error(
        "Found discrepancy in no of Args in encapsulated graph and the "
        "max_index, num_of_inputs/Args " +
        to_string(number_of_inputs) + " size via arg index " + to_string(size));
  }

  m_tensor_manager = make_shared<NGraphTensorManager>(
      GetNgraphClusterName(), GetNgraphClusterId(), GetGraphId(),
      number_of_inputs, number_of_outputs);
}

//---------------------------------------------------------------------------
//  NGraphExecutor::~NGraphExecutor
//---------------------------------------------------------------------------
NGraphExecutor::~NGraphExecutor() {
  auto backend = BackendManager::GetBackend(m_op_backend_name);

  auto destroy_ng_item_callback = std::bind(
      &NGraphExecutor::DestroyCallback, this, std::placeholders::_1, backend);
  m_ng_data_cache.RemoveAll(destroy_ng_item_callback);
  m_tensor_manager.reset();
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
//  NGraphExecutor::GetExecutableFunctionAndTensors
//---------------------------------------------------------------------------
Status NGraphExecutor::GetExecutableFunctionAndTensors(
    const std::vector<Tensor>& tf_input_tensors,
    std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
    std::string& serialized_ng_func, shared_ptr<PipelinedTensorsStore>& pts,
    bool& cache_hit) {
  std::stringstream signature_ss;
  std::vector<TensorShape> input_shapes;
  std::vector<const Tensor*> static_input_map;
  TF_RETURN_IF_ERROR(ComputeSignature(tf_input_tensors, input_shapes,
                                      static_input_map, signature_ss));
  string signature;
  signature = signature_ss.str();

  NGRAPH_VLOG(5) << "Computed signature: " << signature;

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

  // Generate forwarding call to Callback functions
  // CreateCallback and DestroyCallback
  auto create_ng_items_callback =
      std::bind(&NGraphExecutor::CreateCallback, this, std::placeholders::_1,
                input_shapes, static_input_map, op_backend);
  auto destroy_ng_items_callback =
      std::bind(&NGraphExecutor::DestroyCallback, this, std::placeholders::_1,
                op_backend);
  // Get NgItems i.e. ng_executable, serialized ng_functions from Data Cache
  auto status_ng_item_pair =
      m_ng_data_cache.LookUpOrCreate(signature, create_ng_items_callback,
                                     destroy_ng_items_callback, cache_hit);

  if (status_ng_item_pair.first == Status::OK()) {
    std::tie(ng_exec, serialized_ng_func, pts) = status_ng_item_pair.second;
  }
  return status_ng_item_pair.first;
}

//---------------------------------------------------------------------------
//  NGraphExecutor::CallbackCreateItem
//---------------------------------------------------------------------------
std::pair<Status, std::tuple<std::shared_ptr<ngraph::runtime::Executable>,
                             std::string, shared_ptr<PipelinedTensorsStore>>>
NGraphExecutor::CreateCallback(const std::string signature,
                               std::vector<TensorShape> input_shapes,
                               std::vector<const Tensor*> static_input_map,
                               ng::runtime::Backend*& op_backend) {
  std::string serialized_ng_func;
  std::shared_ptr<ngraph::runtime::Executable> ng_exec;
  std::shared_ptr<ngraph::Function> ng_function;
  shared_ptr<PipelinedTensorsStore> pts;
  NGRAPH_VLOG(1) << "Compilation cache miss: " << m_node_name;
  if (!m_do_aot) {
    auto status = Builder::TranslateGraph(input_shapes, static_input_map,
                                          m_graph.get(), ng_function);
    if (status != Status::OK()) {
      return std::make_pair(status,
                            std::make_tuple(ng_exec, serialized_ng_func, pts));
    }
    ng_function->set_friendly_name(m_node_name);
    int json_indentation = 4;
    serialized_ng_func = ngraph::serialize(ng_function, json_indentation);
  } else {
    auto itr = m_aot_functions.find(signature);
    if (itr == m_aot_functions.end()) {
      return std::make_pair(
          errors::Internal(
              "Expected to find AOT precompiled ng function of signature: ",
              signature),
          std::make_tuple(ng_exec, serialized_ng_func, pts));
    }
    serialized_ng_func = itr->second;
  }

  // Serialize to nGraph if needed
  if (std::getenv("NGRAPH_ENABLE_SERIALIZE") != nullptr) {
#if defined NGRAPH_DISTRIBUTED
    int rank_id;
    rank_id = ng::get_distributed_interface()->get_rank();
    auto status = StringToFile(
        "tf_function_" + m_node_name + "_" + to_string(rank_id) + ".json",
        serialized_ng_func);
    if (status != Status::OK()) {
      return std::make_pair(status,
                            std::make_tuple(ng_exec, serialized_ng_func, pts));
    }
#else
    auto status_ser = StringToFile("tf_function_" + m_node_name + ".json",
                                   serialized_ng_func);
    if (status_ser != Status::OK()) {
      return std::make_pair(status_ser,
                            std::make_tuple(ng_exec, serialized_ng_func, pts));
    }
#endif
  }
  // Get NgExecutable
  auto status_ng_exec_pair =
      GetNgExecutable(signature, ng_function, op_backend);
  // Create PipelinedTensorStore
  if (status_ng_exec_pair.first == Status::OK()) {
    ng_exec = status_ng_exec_pair.second;
    auto status_ng_pts_pair = InitializeIOTensorPipeline(
        ng_exec, m_tensor_manager->GetPipelinedInputIndexes(),
        m_tensor_manager->GetPipelinedOutputIndexes());
    pts = status_ng_pts_pair.second;
    return std::make_pair(status_ng_pts_pair.first,
                          std::make_tuple(ng_exec, serialized_ng_func, pts));
  } else {
    Status st = StringToFile("tf_function_error_" + m_node_name + ".json",
                             serialized_ng_func);
    string status_string =
        "Error in compiling op_backend with error: " +
        status_ng_exec_pair.first.error_message() +
        (st.ok() ? "" : (" Also error in dumping serialized function: " +
                         st.error_message()));
    return std::make_pair(errors::Internal(status_string),
                          std::make_tuple(ng_exec, serialized_ng_func, pts));
  }
}

//---------------------------------------------------------------------------
//  NGraphExecutor::GetNgExecutable
//---------------------------------------------------------------------------
std::pair<Status, std::shared_ptr<ngraph::runtime::Executable>>
NGraphExecutor::GetNgExecutable(std::string signature,
                                std::shared_ptr<ngraph::Function>& ng_function,
                                ng::runtime::Backend*& op_backend) {
  std::shared_ptr<ngraph::runtime::Executable> ng_exec;

  ngraph::Event event_compile("Compile nGraph", m_node_name, "");
  BackendManager::LockBackend(m_op_backend_name);
  try {
    if (m_do_aot) {
      auto itr = m_aot_execs.find(signature);
      if (itr == m_aot_execs.end()) {
        BackendManager::UnlockBackend(m_op_backend_name);
        return std::make_pair(
            errors::Internal(
                "Requested AOT, but could not find string with the "
                "signature: ",
                signature),
            nullptr);
      }
      stringstream serialized_exec_read;
      serialized_exec_read << (itr->second);
      ng_exec = op_backend->load(serialized_exec_read);
    } else {
      ng_exec = op_backend->compile(ng_function);
    }
  } catch (const std::exception& exp) {
    BackendManager::UnlockBackend(m_op_backend_name);
    string status_string =
        "Caught exception while compiling op_backend: " + string(exp.what());
    return std::make_pair(errors::Internal(status_string), nullptr);
  } catch (...) {
    BackendManager::UnlockBackend(m_op_backend_name);
    string status_string = "Error in compiling op_backend.";
    return std::make_pair(errors::Internal(status_string), nullptr);
  }
  BackendManager::UnlockBackend(m_op_backend_name);
  event_compile.Stop();
  ngraph::Event::write_trace(event_compile);

  return std::make_pair(Status::OK(), ng_exec);
}

//---------------------------------------------------------------------------
//  NGraphExecutor::DestroyCallback
//---------------------------------------------------------------------------
void NGraphExecutor::DestroyCallback(
    std::tuple<std::shared_ptr<ngraph::runtime::Executable>, std::string,
               shared_ptr<PipelinedTensorsStore>>
        evicted_ng_item,
    ng::runtime::Backend*& op_backend) {
  std::shared_ptr<ngraph::runtime::Executable> evicted_ng_exec;
  std::tie(evicted_ng_exec, std::ignore, std::ignore) = evicted_ng_item;
  // Call delete function here for the erased func
  op_backend->remove_compiled_function(evicted_ng_exec);
  evicted_ng_exec.reset();
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

std::pair<Status, shared_ptr<PipelinedTensorsStore>>
NGraphExecutor::InitializeIOTensorPipeline(
    std::shared_ptr<ngraph::runtime::Executable> ng_exec,
    const vector<int>& pipelined_input_indexes,
    const vector<int>& pipelined_output_indexes) {
  if (!m_executable_can_create_tensor) {
    return std::make_pair(
        errors::Internal(
            "InitializeIOTensorPipeline called, but executable cannot create "
            "tensors"),
        nullptr);
  }
  // Create these pipelined ng tensors only if needed, else reuse from cache
  size_t num_pipelined_inputs = pipelined_input_indexes.size();
  size_t num_pipelined_outputs = pipelined_output_indexes.size();

  // If the input or the output size if 0 then???
  NGRAPH_VLOG(5) << "InitializeIOTensorPipeline: No. of Pipelined Inputs: "
                 << num_pipelined_inputs
                 << " No. of Pipelined Pipelined Outputs: "
                 << num_pipelined_outputs;
  PipelinedTensorMatrix pipelined_input_tensors(m_depth);
  PipelinedTensorMatrix pipelined_output_tensors(m_depth);
  PipelinedTensorVector temp;
  for (size_t i = 0; i < num_pipelined_inputs; i++) {
    int input_index = pipelined_input_indexes[i];
    temp = ng_exec->create_input_tensor(input_index, m_depth);
    for (size_t j = 0; j < temp.size(); j++) {
      pipelined_input_tensors[j].push_back(temp[j]);
    }
  }
  for (size_t i = 0; i < num_pipelined_outputs; i++) {
    int output_index = pipelined_output_indexes[i];
    temp = ng_exec->create_output_tensor(output_index, m_depth);
    for (size_t j = 0; j < temp.size(); j++) {
      pipelined_output_tensors[j].push_back(temp[j]);
    }
  }

  shared_ptr<PipelinedTensorsStore> pts(new PipelinedTensorsStore(
      pipelined_input_tensors, pipelined_output_tensors));
  return std::make_pair(Status::OK(), pts);
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
