/*****************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
*****************************************************************************/

#include "openvino/opsets/opset.hpp"
#include "openvino/pass/convert_fp32_to_fp16.hpp"
#include "openvino/pass/serialize.hpp"

#include "logging/ovtf_log.h"
#include "openvino_tensorflow/default_opset.h"
#include "openvino_tensorflow/executable.h"
#include "openvino_tensorflow/ie_basic_engine.h"
#include "openvino_tensorflow/ie_tensor.h"
#include "openvino_tensorflow/ie_utils.h"
#include "openvino_tensorflow/ie_vadm_engine.h"
#include "openvino_tensorflow/ovtf_utils.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {

Executable::Executable(shared_ptr<ov::Model> model, string device,
                       string device_type)
    : m_device{device},
      m_device_type(device_type),
      m_trivial_fn{nullptr},
      m_model(model) {
  OVTF_VLOG(2) << "Checking for unsupported ops";
  const auto& opset = ov::get_opset8();
  for (const auto& node : model->get_ops()) {
    bool op_supported = false;
    for (const auto& opset : supported_opsets) {
      op_supported = opset.contains_op_type(node.get());
      if (op_supported) break;
    }
    if (!op_supported) {
      OVTF_VLOG(0) << "UNSUPPORTED OP DETECTED: " << node->get_type_info().name;
      throw runtime_error("Detected op " + node->get_name() +
                          " not belonging to opset8!");
    }
  }

  OVTF_VLOG(2) << "Checking for unused parameters";
  auto parameters = model->get_parameters();
  ov::ParameterVector used_parameters;
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
    model = make_shared<ov::Model>(model->get_results(), used_parameters,
                                   model->get_friendly_name());
  }

  // A trivial function is one of
  //  1. constant function (Const -> Result)
  //  2. identity function (Parameter -> Result)
  //  3. zero function (* -> Zero)
  OVTF_VLOG(2) << "Checking for trivial functions";
  bool trivial_fn = true;
  for (auto result : model->get_results()) {
    auto parent = result->input_value(0).get_node_shared_ptr();
    auto pshape = result->get_output_partial_shape(0);
    auto shape = pshape.is_static() ? pshape.to_shape() : ov::Shape{};
    trivial_fn &= ov::is_type<opset::Parameter>(parent) ||
                  ov::is_type<opset::Constant>(parent) ||
                  count(shape.begin(), shape.end(), 0);
  }

  if (trivial_fn) {
    OVTF_VLOG(2) << "Model is trivial and can be short-circuited";
    m_trivial_fn = model;
    return;
  }

  OVTF_VLOG(2) << "Checking for model parameters";
  if (model->get_parameters().size() == 0) {
    OVTF_VLOG(1) << "No parameters found in OpenVINO model!";
    // Try to find a node that can be converted into a "static input"
    bool param_replaced = false;
    for (const auto& node : model->get_ordered_ops()) {
      // Only try to convert constant nodes at the edge to parameters
      // FIXME: IE cannot handle input parameters with i64/u6 precision
      // at the moment
      if (node->get_input_size() == 0 && ov::op::util::is_constant(node) &&
          !(node->get_element_type() == ov::element::i64 ||
            node->get_element_type() == ov::element::u64)) {
        auto constant = ov::as_type_ptr<opset::Constant>(node);
        auto element_type = constant->get_element_type();
        auto shape = constant->get_shape();
        auto param = std::make_shared<opset::Parameter>(element_type, shape);
        param->set_friendly_name(node->get_friendly_name());
        ov::replace_node(node, param);
        // OpenVINO doesn't provide a way to set a parameter to an existing
        // function, so we clone the function here...
        model = make_shared<ov::Model>(model->get_results(),
                                       ov::ParameterVector{param},
                                       model->get_friendly_name());
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
          "Unable to add a parameter to a model with no parameterss");
    }
  }

  m_model = model;

  if (m_device_type == "GPU_FP16") {
    ov::pass::ConvertFP32ToFP16().run_on_model(model);
    model->validate_nodes_and_infer_types();

    auto proc = ov::preprocess::PrePostProcessor(model);
    for (int i = 0; i < model->inputs().size(); i++) {
      if (model->inputs()[i].get_element_type() == ov::element::f16) {
        proc.input(i).tensor().set_element_type(ov::element::f32);
        proc.input(i).preprocess().convert_element_type(ov::element::f16);
      }
    }
    for (int i = 0; i < model->outputs().size(); i++) {
      if (model->outputs()[i].get_element_type() == ov::element::f16) {
        proc.output(i).postprocess().convert_element_type(ov::element::f32);
      }
    }
    model = proc.build();
  }

  if (util::DumpAllGraphs()) {
    std::string name = m_model->get_friendly_name();
    ov::pass::Serialize serializer(name + ".xml", name + ".bin");
    serializer.run_on_model(m_model);
    util::DumpNGGraph(m_model, name + "_executable");
  }

  OVTF_VLOG(2) << "Creating IE Execution Engine";
  if (m_device == "HDDL") {
    m_ie_engine = make_shared<IE_VADM_Engine>(m_model);
  } else {
    m_ie_engine = make_shared<IE_Basic_Engine>(m_model, m_device);
  }
}

bool Executable::Call(const vector<shared_ptr<ov::Tensor>>& inputs,
                      vector<shared_ptr<ov::Tensor>>& outputs,
                      bool multi_req_execution) {
  if (m_trivial_fn) {
    OVTF_VLOG(2) << "Calling trivial function with inputs=" << inputs.size()
                 << " outputs=" << outputs.size();
    return CallTrivial(inputs, outputs);
  }

  auto model = m_ie_engine->get_model();

  // Check if the number of inputs that the OpenVINO model expects is equal to
  // the
  // sum of the
  // inputs specified and the inputs we hoisted, if any.
  if (model->inputs().size() > (inputs.size() + m_hoisted_params.size())) {
    throw runtime_error("Model inputs (" + to_string(model->inputs().size()) +
                        ") number greater than number of given inputs (" +
                        to_string(inputs.size() + m_hoisted_params.size()) +
                        ")");
  }

  //  Prepare input blobs
  auto parameters = model->get_parameters();
  std::vector<std::shared_ptr<IETensor>> ie_inputs(inputs.size());
  std::vector<std::string> input_names(inputs.size());
  for (int i = 0; i < parameters.size(); i++) {
    ov::Any any = parameters[i]->get_rt_info()["index"];
    int64_t input_index = any.as<int64_t>();
    if (find(m_skipped_inputs.begin(), m_skipped_inputs.end(), i) !=
        m_skipped_inputs.end()) {
      continue;
    }
    auto input_name = parameters[i]->get_friendly_name();
    if (m_ie_engine->get_input_idx(input_name) < 0) {
      OVTF_VLOG(1) << "Skipping unused input " << input_name;
      continue;
    }
    ie_inputs[input_index] = static_pointer_cast<IETensor>(inputs[input_index]);
    input_names[input_index] = input_name;
  }

  std::vector<std::shared_ptr<IETensor>> ie_hoisted_params(
      m_hoisted_params.size());
  std::vector<std::string> param_names(m_hoisted_params.size());
  int j = 0;
  for (const auto& it : m_hoisted_params) {
    auto input_name = it.first;
    if (m_ie_engine->get_input_idx(input_name) < 0) {
      OVTF_VLOG(1) << "Skipping unused hoisted param " << input_name;
      continue;
    }
    ie_hoisted_params[j] = nullptr;
    ie_hoisted_params[j] = static_pointer_cast<IETensor>(it.second);
    param_names[j++] = input_name;
  }

  if (outputs.size() == 0 && model->outputs().size() > 0) {
    outputs.resize(model->outputs().size(), nullptr);
  }

  //  Prepare output blobs
  auto results = model->get_results();
  std::vector<std::shared_ptr<IETensor>> ie_outputs(outputs.size());
  std::vector<std::string> output_names(outputs.size());
  for (int i = 0; i < results.size(); i++) {
    if (outputs[i] != nullptr) {
      ie_outputs[i] = static_pointer_cast<IETensor>(outputs[i]);
    }
    output_names[i] = results[i]->get_friendly_name();
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

bool Executable::CallTrivial(const vector<shared_ptr<ov::Tensor>>& inputs,
                             vector<shared_ptr<ov::Tensor>>& outputs) {
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
      OVTF_VLOG(2) << "Skipping model with zero dim result...";
      continue;
    }
    auto parent = results[i]->input_value(0).get_node_shared_ptr();
    if (ov::is_type<opset::Parameter>(parent)) {
      OVTF_VLOG(2) << "Calling parameter -> result function...";
      auto param = ov::as_type_ptr<opset::Parameter>(parent);
      auto index = m_trivial_fn->get_parameter_index(param);
      if (index < 0) {
        throw runtime_error("Input parameter " + param->get_friendly_name() +
                            " not found in trivial function");
      }
      if (outputs[i] == nullptr) {
        outputs[i] = make_shared<IETensor>(inputs[index]->get_element_type(),
                                           inputs[index]->get_shape());
      }
      auto size = inputs[index]->get_byte_size();
      unsigned char* buf_ptr = new unsigned char[size];
      std::copy((uint8_t*)(inputs[index]->data()),
                ((uint8_t*)(inputs[index]->data())) + size, buf_ptr);
      std::copy((uint8_t*)buf_ptr, ((uint8_t*)buf_ptr) + size,
                (uint8_t*)(outputs[i]->data()));
      delete[] buf_ptr;
    } else if (ov::is_type<opset::Constant>(parent)) {
      OVTF_VLOG(2) << "Calling constant -> result function...";
      auto constant = ov::as_type_ptr<opset::Constant>(parent);
      if (outputs[i] == nullptr) {
        outputs[i] = make_shared<IETensor>(
            constant->get_element_type(), constant->get_shape(),
            const_cast<void*>(constant->get_data_ptr()));
      } else {
        std::copy((uint8_t*)constant->get_data_ptr(),
                  ((uint8_t*)(constant->get_data_ptr())) +
                      (shape_size(constant->get_shape()) *
                       constant->get_element_type().size()),
                  (uint8_t*)(outputs[i]->data()));
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
  if (!m_model || !m_ie_engine) return;
  std::string name = m_model->get_friendly_name();
  ov::pass::Serialize serializer(output_dir + "/" + name + ".xml",
                                 output_dir + "/" + name + ".bin");
  serializer.run_on_model(m_model);
}
}  // namespace openvino_tensorflow
}  // namespace tensorflow
