//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset.hpp"

#include "logging/ngraph_log.h"
#include "ngraph_bridge/default_opset.h"
#include "ngraph_bridge/ie_executable.h"
#include "ngraph_bridge/ie_tensor.h"

using namespace std;
using namespace ngraph;

namespace tensorflow {
namespace ngraph_bridge {

IE_Executable::IE_Executable(shared_ptr<Function> func, string device)
    : m_device{device}, m_trivial_fn{nullptr} {
  NGRAPH_VLOG(2) << "Checking for unsupported ops in IE backend";
  const auto& opset = ngraph::get_opset3();
  for (const auto& node : func->get_ops()) {
    if (!opset.contains_op_type(node.get())) {
      NGRAPH_VLOG(0) << "UNSUPPORTED OP DETECTED: "
                     << node->get_type_info().name;
      THROW_IE_EXCEPTION << "Detected op not belonging to opset3!";
    }
  }

  NGRAPH_VLOG(2) << "Checking for trivial functions in IE backend";
  bool trivial_fn = true;
  for (auto result : func->get_results()) {
    auto parent = result->input_value(0).get_node_shared_ptr();
    trivial_fn &= ngraph::is_type<opset::Parameter>(parent) ||
                  ngraph::is_type<opset::Constant>(parent);
  }

  if (trivial_fn) {
    NGRAPH_VLOG(2) << "Function is trivial and can be short-circuited";
    set_parameters_and_results(*func);
    m_trivial_fn = func;
    return;
  }

  NGRAPH_VLOG(2) << "Checking for function parameters in IE backend";
  if (func->get_parameters().size() == 0) {
    NGRAPH_VLOG(1) << "No parameters found in nGraph function!";
    // Try to find a node that can be converted into a "static input"
    bool param_replaced = false;
    for (const auto& node : func->get_ordered_ops()) {
      // Only try to convert constant nodes at the edge to parameters
      // FIXME: IE cannot handle input parameters with i64/u6 precision
      // at the moment
      if (node->get_input_size() == 0 && node->is_constant() &&
          !(node->get_element_type() == ngraph::element::i64 ||
            node->get_element_type() == ngraph::element::u64)) {
        auto constant = ngraph::as_type_ptr<opset::Constant>(node);
        auto element_type = constant->get_element_type();
        auto shape = constant->get_shape();
        auto param = std::make_shared<opset::Parameter>(element_type, shape);
        param->set_friendly_name(node->get_friendly_name());
        ngraph::replace_node(node, param);
        // nGraph doesn't provide a way to set a parameter to an existing
        // function, so we clone the function here...
        func = make_shared<Function>(func->get_results(),
                                     ParameterVector{param}, func->get_name());
        auto ie_tensor = make_shared<IETensor>(element_type, shape);
        ie_tensor->write(constant->get_data_ptr(),
                         shape_size(shape) * element_type.size());
        m_hoisted_params.push_back(
            make_pair(param->get_friendly_name(), ie_tensor));
        NGRAPH_VLOG(1) << "Converted node " << constant << " to a parameter "
                       << param;
        param_replaced = true;
        break;
      }
      if (!param_replaced) {
        THROW_IE_EXCEPTION
            << "Unable to add a parameter to a function with no parameters!";
      }
    }
  }

  set_parameters_and_results(*func);

  NGRAPH_VLOG(2) << "Creating IE CNN network using nGraph function";
  m_network = InferenceEngine::CNNNetwork(func);

  if (std::getenv("NGRAPH_TF_DUMP_GRAPHS")) {
    auto& name = m_network.getName();
    m_network.serialize(name + ".xml", name + ".bin");
  }

  NGRAPH_VLOG(2) << "Loading IE CNN network to device " << m_device;

  InferenceEngine::Core ie;
  // Load network to the plugin (m_device) and create an infer request
  InferenceEngine::ExecutableNetwork exe_network =
      ie.LoadNetwork(m_network, m_device);
  m_infer_req = exe_network.CreateInferRequest();
}

bool IE_Executable::call(const vector<shared_ptr<runtime::Tensor>>& outputs,
                         const vector<shared_ptr<runtime::Tensor>>& inputs) {
  if (m_trivial_fn) {
    NGRAPH_VLOG(2) << "Calling trivial IE function with inputs="
                   << inputs.size() << " outputs=" << outputs.size();
    return call_trivial(outputs, inputs);
  }

  // Check if the number of inputs that the CNN network expects is equal to the
  // sum of the
  // inputs specified and the inputs we hoisted, if any.
  InferenceEngine::InputsDataMap input_info = m_network.getInputsInfo();
  if (input_info.size() != (inputs.size() + m_hoisted_params.size())) {
    THROW_IE_EXCEPTION
        << "Function inputs number differ from number of given inputs";
  }

  //  Prepare input blobs
  auto func = m_network.getFunction();
  auto parameters = func->get_parameters();
  for (int i = 0; i < inputs.size(); i++) {
    shared_ptr<IETensor> tv = static_pointer_cast<IETensor>(inputs[i]);
    m_infer_req.SetBlob(parameters[i]->get_friendly_name(), tv->get_blob());
  }

  for (const auto& it : m_hoisted_params) {
    shared_ptr<IETensor> tv = static_pointer_cast<IETensor>(it.second);
    m_infer_req.SetBlob(it.first, tv->get_blob());
  }

  InferenceEngine::OutputsDataMap output_info = m_network.getOutputsInfo();
  if (output_info.size() != outputs.size()) {
    THROW_IE_EXCEPTION
        << "Function outputs number differ from number of given outputs";
  }

  //  Prepare output blobs
  auto results = func->get_results();
  for (int i = 0; i < outputs.size(); i++) {
    shared_ptr<IETensor> tv = static_pointer_cast<IETensor>(outputs[i]);
    // Since IE has no "result" nodes, we set the blob corresponding to the
    // parent of this result node
    auto parent = results[i]->input_value(0).get_node_shared_ptr();
    m_infer_req.SetBlob(parent->get_friendly_name(), tv->get_blob());
  }

  m_infer_req.Infer();
  return true;
}

bool IE_Executable::call_trivial(
    const vector<shared_ptr<runtime::Tensor>>& outputs,
    const vector<shared_ptr<runtime::Tensor>>& inputs) {
  // outputs are in the same order as results
  auto results = m_trivial_fn->get_results();
  for (int i = 0; i < outputs.size(); i++) {
    auto parent = results[i]->input_value(0).get_node_shared_ptr();
    if (ngraph::is_type<opset::Parameter>(parent)) {
      auto param = ngraph::as_type_ptr<opset::Parameter>(parent);
      auto index = m_trivial_fn->get_parameter_index(param);
      if (index < 0) {
        THROW_IE_EXCEPTION << "Input parameter " << param->get_friendly_name()
                           << " not found in trivial function";
      }
      auto size = inputs[index]->get_size_in_bytes();
      unsigned char* buf_ptr = new unsigned char[size];
      inputs[index]->read(buf_ptr, size);
      outputs[0]->write(buf_ptr, size);
      delete buf_ptr;
    } else if (ngraph::is_type<opset::Constant>(parent)) {
      auto constant = ngraph::as_type_ptr<opset::Constant>(parent);
      outputs[i]->write(constant->get_data_ptr(),
                        shape_size(constant->get_shape()) *
                            constant->get_element_type().size());
    } else {
      THROW_IE_EXCEPTION << "Expected constant or parameter feeding to a "
                            "result in trivial function";
    }
  }
  return true;
}
}
}
