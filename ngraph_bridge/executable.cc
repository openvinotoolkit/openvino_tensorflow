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

#include <ie_plugin_config.hpp>

#include "logging/ngraph_log.h"
#include "ngraph_bridge/default_opset.h"
#include "ngraph_bridge/executable.h"
#include "ngraph_bridge/ie_tensor.h"
#include "ngraph_bridge/ngraph_utils.h"

using namespace std;
using namespace ngraph;

namespace tensorflow {
namespace ngraph_bridge {

Executable::Executable(shared_ptr<Function> func, string device)
    : m_device{device}, m_trivial_fn{nullptr}, m_function(func) {
  NGRAPH_VLOG(2) << "Checking for unsupported ops";
  const auto& opset = ngraph::get_opset5();
  for (const auto& node : func->get_ops()) {
    if (!opset.contains_op_type(node.get())) {
      NGRAPH_VLOG(0) << "UNSUPPORTED OP DETECTED: "
                     << node->get_type_info().name;
      throw runtime_error("Detected op " + node->get_name() +
                          " not belonging to opset5!");
    }
  }

  NGRAPH_VLOG(2) << "Checking for unused parameters";
  auto parameters = func->get_parameters();
  ngraph::ParameterVector used_parameters;
  for (int i = 0; i < parameters.size(); ++i) {
    NGRAPH_VLOG(3) << parameters[i];
    if (parameters[i]->get_users().size() == 0) {
      m_skipped_inputs.push_back(i);
      NGRAPH_VLOG(2) << "Removing unused parameter "
                     << parameters[i]->get_name();
    } else {
      used_parameters.push_back(parameters[i]);
    }
  }
  if (parameters.size() != used_parameters.size()) {
    func = make_shared<Function>(func->get_results(), used_parameters,
                                 func->get_friendly_name());
  }

  // A trivial function is one of
  //  1. constant function (Const -> Result)
  //  2. identity function (Parameter -> Result)
  //  3. zero function (* -> Zero)
  NGRAPH_VLOG(2) << "Checking for trivial functions";
  bool trivial_fn = true;
  for (auto result : func->get_results()) {
    auto parent = result->input_value(0).get_node_shared_ptr();
    auto pshape = result->get_output_partial_shape(0);
    auto shape = pshape.is_static() ? pshape.to_shape() : Shape{};
    trivial_fn &= ngraph::is_type<opset::Parameter>(parent) ||
                  ngraph::is_type<opset::Constant>(parent) ||
                  count(shape.begin(), shape.end(), 0);
  }

  if (trivial_fn) {
    NGRAPH_VLOG(2) << "Function is trivial and can be short-circuited";
    m_trivial_fn = func;
    return;
  }

  NGRAPH_VLOG(2) << "Checking for function parameters";
  if (func->get_parameters().size() == 0) {
    NGRAPH_VLOG(1) << "No parameters found in nGraph function!";
    // Try to find a node that can be converted into a "static input"
    bool param_replaced = false;
    for (const auto& node : func->get_ordered_ops()) {
      // Only try to convert constant nodes at the edge to parameters
      // FIXME: IE cannot handle input parameters with i64/u6 precision
      // at the moment
      if (node->get_input_size() == 0 && ngraph::op::is_constant(node) &&
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
        func =
            make_shared<Function>(func->get_results(), ParameterVector{param},
                                  func->get_friendly_name());
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
    }
    if (!param_replaced) {
      throw runtime_error(
          "Unable to add a parameter to a function with no parameterss");
    }
  }

  m_function = func;

  NGRAPH_VLOG(2) << "Creating IE CNN network using nGraph function";
  m_network = InferenceEngine::CNNNetwork(func);

  InferenceEngine::Core ie;
  std::map<string, string> options;

  if (util::DumpAllGraphs()) {
    auto& name = m_function->get_friendly_name();
    m_network.serialize(name + ".xml", name + ".bin");
    util::DumpNGGraph(func, name + "_executable");
    options[InferenceEngine::PluginConfigParams::KEY_DUMP_EXEC_GRAPH_AS_DOT] =
        name + "_IE_" + m_device;
  }

  NGRAPH_VLOG(2) << "Loading IE CNN network to device " << m_device;

  // Load network to the plugin (m_device) and create an infer request
  InferenceEngine::ExecutableNetwork exe_network =
      ie.LoadNetwork(m_network, m_device, options);
  m_infer_req = exe_network.CreateInferRequest();
}

bool Executable::Call(const vector<shared_ptr<runtime::Tensor>>& inputs,
                      vector<shared_ptr<runtime::Tensor>>& outputs) {
  if (m_trivial_fn) {
    NGRAPH_VLOG(2) << "Calling trivial IE function with inputs="
                   << inputs.size() << " outputs=" << outputs.size();
    return CallTrivial(inputs, outputs);
  }

  // Check if the number of inputs that the CNN network expects is equal to the
  // sum of the
  // inputs specified and the inputs we hoisted, if any.
  InferenceEngine::InputsDataMap input_info = m_network.getInputsInfo();
  if (input_info.size() > (inputs.size() + m_hoisted_params.size())) {
    throw runtime_error("Function inputs (" + to_string(input_info.size()) +
                        ") number greater than number of given inputs (" +
                        to_string(inputs.size() + m_hoisted_params.size()) +
                        ")");
  }

  //  Prepare input blobs
  auto func = m_network.getFunction();
  auto parameters = func->get_parameters();
  int j = 0;
  for (int i = 0; i < inputs.size(); i++) {
    if (find(m_skipped_inputs.begin(), m_skipped_inputs.end(), i) !=
        m_skipped_inputs.end()) {
      continue;
    }
    auto input_name = parameters[j++]->get_friendly_name();
    if (input_info.find(input_name) == input_info.end()) {
      NGRAPH_VLOG(1) << "Skipping unused input " << input_name;
      continue;
    }
    shared_ptr<IETensor> tv = static_pointer_cast<IETensor>(inputs[i]);
    m_infer_req.SetBlob(input_name, tv->get_blob());
  }

  for (const auto& it : m_hoisted_params) {
    auto input_name = it.first;
    if (input_info.find(input_name) == input_info.end()) {
      NGRAPH_VLOG(1) << "Skipping unused hoisted param " << input_name;
      continue;
    }
    shared_ptr<IETensor> tv = static_pointer_cast<IETensor>(it.second);
    m_infer_req.SetBlob(input_name, tv->get_blob());
  }

  InferenceEngine::OutputsDataMap output_info = m_network.getOutputsInfo();
  if (outputs.size() == 0 && output_info.size() > 0) {
    outputs.resize(output_info.size(), nullptr);
  }

  auto get_output_name = [](std::shared_ptr<ngraph::Node> node) {
    // Since IE has no "result" nodes, we set the blob corresponding to the
    // parent of this result node
    auto parent = node->input_value(0).get_node_shared_ptr();
    auto name = parent->get_friendly_name();
    // if parent has multiple outputs, correctly identify the output feeding
    // into this result
    if (parent->outputs().size() > 1) {
      name += "." + to_string(node->input_value(0).get_index());
    }
    return name;
  };

  //  Prepare output blobs
  auto results = func->get_results();
  for (int i = 0; i < results.size(); i++) {
    if (outputs[i] != nullptr) {
      NGRAPH_VLOG(4) << "Executable::call() SetBlob()";
      shared_ptr<IETensor> tv = static_pointer_cast<IETensor>(outputs[i]);
      m_infer_req.SetBlob(get_output_name(results[i]), tv->get_blob());
    }
  }

  m_infer_req.Infer();

  // Set dynamic output blobs
  for (int i = 0; i < results.size(); i++) {
    if (outputs[i] == nullptr) {
      NGRAPH_VLOG(4) << "Executable::call() GetBlob()";
      auto blob = m_infer_req.GetBlob(get_output_name(results[i]));
      outputs[i] = make_shared<IETensor>(blob);
    }
  }

  return true;
}

bool Executable::CallTrivial(const vector<shared_ptr<runtime::Tensor>>& inputs,
                             vector<shared_ptr<runtime::Tensor>>& outputs) {
  // outputs are in the same order as results
  auto results = m_trivial_fn->get_results();
  if (outputs.size() == 0 && results.size() > 0) {
    outputs.resize(results.size(), nullptr);
  }

  for (int i = 0; i < results.size(); i++) {
    auto& shape = results[i]->get_shape();
    if (count(shape.begin(), shape.end(), 0)) {
      if (outputs[i] == nullptr) {
        outputs[i] =
            make_shared<IETensor>(results[i]->get_element_type(), shape);
      }
      NGRAPH_VLOG(2) << "Skipping function with zero dim result...";
      continue;
    }
    auto parent = results[i]->input_value(0).get_node_shared_ptr();
    if (ngraph::is_type<opset::Parameter>(parent)) {
      NGRAPH_VLOG(2) << "Calling parameter -> result function...";
      auto param = ngraph::as_type_ptr<opset::Parameter>(parent);
      auto index = m_trivial_fn->get_parameter_index(param);
      if (index < 0) {
        throw runtime_error("Input parameter " + param->get_friendly_name() +
                            " not found in trivial function");
      }
      if (outputs[i] == nullptr) {
        outputs[i] = make_shared<IETensor>(inputs[index]->get_element_type(),
                                           inputs[index]->get_shape());
      }
      auto size = inputs[index]->get_size_in_bytes();
      unsigned char* buf_ptr = new unsigned char[size];
      inputs[index]->read(buf_ptr, size);
      outputs[i]->write(buf_ptr, size);
      delete buf_ptr;
    } else if (ngraph::is_type<opset::Constant>(parent)) {
      NGRAPH_VLOG(2) << "Calling constant -> result function...";
      auto constant = ngraph::as_type_ptr<opset::Constant>(parent);
      if (outputs[i] == nullptr) {
        outputs[i] = make_shared<IETensor>(
            constant->get_element_type(), constant->get_shape(),
            const_cast<void*>(constant->get_data_ptr()));
      } else {
        outputs[i]->write(constant->get_data_ptr(),
                          shape_size(constant->get_shape()) *
                              constant->get_element_type().size());
      }
    } else {
      throw runtime_error(
          "Expected constant or parameter feeding to a "
          "result in trivial function");
    }
  }
  return true;
}
}
}
