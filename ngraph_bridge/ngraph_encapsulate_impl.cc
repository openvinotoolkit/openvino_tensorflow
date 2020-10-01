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

// Ngraph Encapsulate Implementation class for EncapsulateOp class
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

// Calls ComputeSignature and gets ngraph executable
Status NGraphEncapsulateImpl::GetNgExecutable(
    const std::vector<Tensor>& tf_input_tensors,
    std::vector<TensorShape>& input_shapes,
    std::vector<const Tensor*>& static_input_map,
    std::shared_ptr<Executable>& ng_exec,
    std::shared_ptr<ngraph::Function>& ng_function) {
  auto backend = BackendManager::GetBackend();

  // Compute Signature
  std::stringstream signature_ss;
  TF_RETURN_IF_ERROR(ComputeSignature(tf_input_tensors, input_shapes,
                                      static_input_map, signature_ss));
  string signature = signature_ss.str();
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
    TF_RETURN_IF_ERROR(Builder::TranslateGraph(input_shapes, static_input_map,
                                               &m_graph, ng_function));
    ng_function->set_friendly_name(m_name);

    if (std::getenv("NGRAPH_TF_DUMP_GRAPHS") != nullptr) {
      ngraph::plot_graph(ng_function, "tf_function_" + m_name + ".dot");
    }

    // Evict the cache if the number of elements exceeds the limit
    std::shared_ptr<Executable> evicted_ng_exec;
    const char* cache_depth_specified =
        std::getenv("NGRAPH_TF_FUNCTION_CACHE_ITEM_DEPTH");
    if (cache_depth_specified != nullptr) {
      m_function_cache_depth_in_items = atoi(cache_depth_specified);
    }
    if (m_ng_exec_map.size() >= m_function_cache_depth_in_items) {
      evicted_ng_exec = m_ng_exec_map[m_lru.back()];
      m_ng_exec_map.erase(m_lru.back());

      // Call delete function here for the erased func
      backend->remove_compiled_function(evicted_ng_exec);

      m_lru.pop_back();
    }  // cache eviction if cache size greater than cache depth

    NG_TRACE("Compile nGraph", m_name, "");
    try {
      ng_exec = backend->compile(ng_function);
    } catch (const std::exception& ex) {
      return errors::Internal("Failed to compile function " + m_name + ": ",
                              ex.what());
    }

    SetNgExecMap(signature, ng_exec);
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

Status NGraphEncapsulateImpl::AllocateNGTensors(
    const std::vector<Tensor>& tf_tensors,
    vector<shared_ptr<ngraph::runtime::Tensor>>& ng_tensors) {
  for (int i = 0; i < tf_tensors.size(); i++) {
    ngraph::Shape ng_shape(tf_tensors[i].shape().dims());
    for (int j = 0; j < tf_tensors[i].shape().dims(); ++j) {
      ng_shape[j] = tf_tensors[i].shape().dim_size(j);
    }
    ngraph::element::Type ng_element_type;
    TF_RETURN_IF_ERROR(
        TFDataTypeToNGraphElementType(tf_tensors[i].dtype(), &ng_element_type));

    auto backend = BackendManager::GetBackend();
    std::shared_ptr<ngraph::runtime::Tensor> ng_tensor =
        backend->create_tensor(ng_element_type, ng_shape, tf_tensors[i].data());
    ng_tensors.push_back(ng_tensor);
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
      // right now _ngraph_ is used for optional attributes
      auto attr_name = itx.first;
      auto attr_value = itx.second.s();
      NGRAPH_VLOG(4) << "Attribute: " << attr_name.substr(strlen("_ngraph_"))
                     << " Value: " << attr_value;
      additional_attribute_map->insert(
          {attr_name.substr(strlen("_ngraph_")), attr_value});
    }
  }
  return Status::OK();
}

void NGraphEncapsulateImpl::NGraphEncapsulateImpl::ClearExecMaps() {
  m_ng_exec_map.clear();
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
