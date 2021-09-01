/*****************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
*****************************************************************************/

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset.hpp"
#include "ngraph/pass/convert_fp32_to_fp16.hpp"

#include <ie_plugin_config.hpp>

#include "logging/ovtf_log.h"
#include "openvino_tensorflow/default_opset.h"
#include "openvino_tensorflow/executable.h"
#include "openvino_tensorflow/ie_basic_engine.h"
#include "openvino_tensorflow/ie_tensor.h"
#include "openvino_tensorflow/ie_utils.h"
#include "openvino_tensorflow/ie_vadm_engine.h"
#include "openvino_tensorflow/ovtf_utils.h"

using namespace std;
using namespace ngraph;

namespace tensorflow {
namespace openvino_tensorflow {

Executable::Executable(shared_ptr<Function> func, string device,
                       string device_type)
    : m_device{device},
      m_device_type(device_type),
      m_trivial_fn{nullptr},
      m_function(func) {
  OVTF_VLOG(2) << "Checking for unsupported ops";
  const auto& opset = ngraph::get_opset5();
  for (const auto& node : func->get_ops()) {
    if (!opset.contains_op_type(node.get())) {
      OVTF_VLOG(0) << "UNSUPPORTED OP DETECTED: " << node->get_type_info().name;
      throw runtime_error("Detected op " + node->get_name() +
                          " not belonging to opset5!");
    }
  }

  OVTF_VLOG(2) << "Checking for unused parameters";
  auto parameters = func->get_parameters();
  ngraph::ParameterVector used_parameters;
  for (int i = 0; i < parameters.size(); ++i) {
    OVTF_VLOG(3) << parameters[i];
    if (parameters[i]->get_users().size() == 0) {
      m_skipped_inputs.push_back(i);
      OVTF_VLOG(2) << "Removing unused parameter " << parameters[i]->get_name();
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
  OVTF_VLOG(2) << "Checking for trivial functions";
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
    OVTF_VLOG(2) << "Function is trivial and can be short-circuited";
    m_trivial_fn = func;
    return;
  }

  OVTF_VLOG(2) << "Checking for function parameters";
  if (func->get_parameters().size() == 0) {
    OVTF_VLOG(1) << "No parameters found in nGraph function!";
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
        OVTF_VLOG(1) << "Converted node " << constant << " to a parameter "
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

  if (m_device_type == "GPU_FP16") {
    ngraph::pass::ConvertFP32ToFP16().run_on_function(func);
    func->validate_nodes_and_infer_types();
  }

  OVTF_VLOG(2) << "Creating IE CNN network using nGraph function";
  m_network = InferenceEngine::CNNNetwork(func);

  std::map<string, string> options;

  if (util::DumpAllGraphs()) {
    auto& name = m_function->get_friendly_name();
    m_network.serialize(name + ".xml", name + ".bin");
    util::DumpNGGraph(func, name + "_executable");
    options[InferenceEngine::PluginConfigParams::KEY_DUMP_EXEC_GRAPH_AS_DOT] =
        name + "_IE_" + m_device;
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

  std::unordered_map<std::string, element::Type> output_dt_map;

  auto results = func->get_results();
  for (int i = 0; i < results.size(); i++) {
    auto output_name = get_output_name(results[i]);
    auto dtype = results[i]->get_element_type();
    output_dt_map[output_name] = dtype;
  }

  auto outputInfo = m_network.getOutputsInfo();
  for (auto iter = outputInfo.begin(); iter != outputInfo.end(); ++iter) {
    auto out_name = iter->first;
    auto it = output_dt_map.find(out_name);

    if (it == output_dt_map.end()) {
      THROW_IE_EXCEPTION << "Output Mismatch: Output " << out_name
                         << " doesn't exist";
    }
    auto precision = IE_Utils::toPrecision(it->second);
    if (m_device_type == "GPU_FP16") {
      precision = InferenceEngine::Precision::FP32;
    }
    iter->second->setPrecision(precision);
  }

  OVTF_VLOG(2) << "Creating IE Execution Engine";
  if (m_device == "HDDL") {
    m_ie_engine = make_shared<IE_VADM_Engine>(m_network);
  } else {
    m_ie_engine = make_shared<IE_Basic_Engine>(m_network, m_device);
  }
}

bool Executable::Call(const vector<shared_ptr<runtime::Tensor>>& inputs,
                      vector<shared_ptr<runtime::Tensor>>& outputs,
                      bool multi_req_execution) {
  if (m_trivial_fn) {
    OVTF_VLOG(2) << "Calling trivial IE function with inputs=" << inputs.size()
                 << " outputs=" << outputs.size();
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
  auto func = m_ie_engine->get_func();
  auto parameters = func->get_parameters();
  std::vector<std::shared_ptr<IETensor>> ie_inputs(inputs.size());
  std::vector<std::string> input_names(inputs.size());
  int j = 0;
  for (int i = 0; i < inputs.size(); i++) {
    if (find(m_skipped_inputs.begin(), m_skipped_inputs.end(), i) !=
        m_skipped_inputs.end()) {
      continue;
    }
    auto input_name = parameters[j++]->get_friendly_name();
    if (input_info.find(input_name) == input_info.end()) {
      OVTF_VLOG(1) << "Skipping unused input " << input_name;
      continue;
    }
    ie_inputs[i] = nullptr;
    ie_inputs[i] = static_pointer_cast<IETensor>(inputs[i]);
    input_names[i] = input_name;
  }

  std::vector<std::shared_ptr<IETensor>> ie_hoisted_params(
      m_hoisted_params.size());
  std::vector<std::string> param_names(m_hoisted_params.size());
  for (const auto& it : m_hoisted_params) {
    auto input_name = it.first;
    if (input_info.find(input_name) == input_info.end()) {
      OVTF_VLOG(1) << "Skipping unused hoisted param " << input_name;
      continue;
    }
    ie_hoisted_params[j] = nullptr;
    ie_hoisted_params[j] = static_pointer_cast<IETensor>(it.second);
    param_names[j++] = input_name;
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
  std::vector<std::shared_ptr<IETensor>> ie_outputs(outputs.size());
  std::vector<std::string> output_names(outputs.size());
  for (int i = 0; i < results.size(); i++) {
    if (outputs[i] != nullptr) {
      ie_outputs[i] = static_pointer_cast<IETensor>(outputs[i]);
    }
    output_names[i] = get_output_name(results[i]);
  }

  if (multi_req_execution) {
    m_ie_engine->enable_multi_req_execution();
  }

  m_ie_engine->infer(ie_inputs, input_names, ie_outputs, output_names,
                     ie_hoisted_params, param_names);

  // Set dynamic output blobs
  for (int i = 0; i < results.size(); i++) {
    if (outputs[i] == nullptr) {
      outputs[i] = ie_outputs[i];
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
      OVTF_VLOG(2) << "Skipping function with zero dim result...";
      continue;
    }
    auto parent = results[i]->input_value(0).get_node_shared_ptr();
    if (ngraph::is_type<opset::Parameter>(parent)) {
      OVTF_VLOG(2) << "Calling parameter -> result function...";
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
      delete[] buf_ptr;
    } else if (ngraph::is_type<opset::Constant>(parent)) {
      OVTF_VLOG(2) << "Calling constant -> result function...";
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

void Executable::ExportIR(const string& output_dir) {
  if (!m_function || !m_ie_engine) return;
  auto& name = m_function->get_friendly_name();
  m_network.serialize(output_dir + "/" + name + ".xml",
                      output_dir + "/" + name + ".bin");
}
}  // namespace openvino_tensorflow
}  // namespace tensorflow
