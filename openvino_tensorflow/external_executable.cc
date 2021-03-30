/*****************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
*****************************************************************************/

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset.hpp"

#include <ie_plugin_config.hpp>

#include "logging/ovtf_log.h"
#include "openvino_tensorflow/default_opset.h"
#include "openvino_tensorflow/executable.h"
#include "openvino_tensorflow/ie_tensor.h"
#include "openvino_tensorflow/ovtf_utils.h"
#include "openvino_tensorflow/ie_basic_engine.h"
#include "openvino_tensorflow/ie_utils.h"
#include "openvino_tensorflow/ie_vadm_engine.h"

using namespace std;
using namespace ngraph;

namespace tensorflow {
namespace openvino_tensorflow {

//ExternalExecutable::ExternalExecutable(shared_ptr<Function> func, string device)
//    : m_device{device}, m_trivial_fn{nullptr}, m_function(func) {
ExternalExecutable::ExternalExecutable(string ir_path, string device)
    : m_device{device}, m_trivial_fn{nullptr} {

  OVTF_VLOG(2) << "Creating IE CNN network using nGraph function";
  //m_network = InferenceEngine::CNNNetwork(func);
  InferenceEngine::Core ie_core;
  m_network = ie_core.ReadNetwork(ir_path+".xml", ir_path+".bin");

  std::map<string, string> options;

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

  auto results = m_network.getFunction()->get_results();
  for (int i = 0; i < results.size(); i++) {
    auto output_name = get_output_name(results[i]);
    auto dtype = results[i]->get_element_type();
    output_dt_map[output_name] = dtype;
  }

  auto outputInfo = m_network.getOutputsInfo();
  for (auto iter = outputInfo.begin(); iter != outputInfo.end(); ++iter){
    auto out_name = iter->first;
    auto it = output_dt_map.find(out_name);

    if(it == output_dt_map.end()){

      THROW_IE_EXCEPTION << "Output Mismatch: Output " << out_name << " doesn't exist";
    }
    auto precision = IE_Utils::toPrecision(it->second);
    iter->second->setPrecision(precision);
  }

  OVTF_VLOG(2) << "Creating IE Execution Engine";
  if (m_device == "HDDL") {
    m_ie_engine = make_shared<IE_VADM_Engine>(m_network);
  } else {
    m_ie_engine = make_shared<IE_Basic_Engine>(m_network, m_device);
  }
}

bool ExternalExecutable::Call(ExternalTensor* inputs,
                      ExternalTensor* params,
                      ExternalTensor* outputs,
                      size_t num_inputs,
                      size_t num_params,
                      size_t num_outputs,
                      bool multi_req_execution) {

  std::vector<std::shared_ptr<IETensor>> ie_inputs(num_inputs);
  std::vector<std::string> input_names(num_inputs);
  for (int i=0; i<num_inputs; i++) {
    input_names[i] = std::string(inputs[i].name);
    if (inputs[i].valid) {
      ngraph::element::Type element_type;
      switch (inputs[i].type) {
        case OVTF_DATA_TYPE::OVTF_FP32:
          element_type =  element::Type_t::f32;
        case OVTF_DATA_TYPE::OVTF_I8:
          element_type =  element::Type_t::i8;
        case OVTF_DATA_TYPE::OVTF_U8:
          element_type =  element::Type_t::u8;
        case OVTF_DATA_TYPE::OVTF_I16:
          element_type =  element::Type_t::i16;
        case OVTF_DATA_TYPE::OVTF_U16:
          element_type =  element::Type_t::u16;
        case OVTF_DATA_TYPE::OVTF_I32:
          element_type =  element::Type_t::i32;
        case OVTF_DATA_TYPE::OVTF_I64:
          element_type =  element::Type_t::i64;
        case OVTF_DATA_TYPE::OVTF_U64:
          element_type =  element::Type_t::u64;
        case OVTF_DATA_TYPE::OVTF_BOOL:
          element_type =  element::Type_t::boolean;
        default:
          THROW_IE_EXCEPTION << "Can't convert type OVTF Data Type to nGraph Element Type!";
      }
      ngraph::Shape ng_shape(inputs[i].num_dims);
      for (int j=0; j<inputs[i].num_dims; j++)
        ng_shape[j] = inputs[i].dims[j];
      ie_inputs[i] = std::make_shared<IETensor>(element_type, ng_shape, inputs[i].memory_pointer);
    } else {
        ie_inputs[i] = nullptr;
    }
  }

  std::vector<std::shared_ptr<IETensor>> ie_params(num_params);
  std::vector<std::string> param_names(num_params);
  for (int i=0; i<num_params; i++) {
    param_names[i] = std::string(params[i].name);
    if (params[i].valid) {
      ngraph::element::Type element_type;
      switch (params[i].type) {
        case OVTF_DATA_TYPE::OVTF_FP32:
          element_type =  element::Type_t::f32;
        case OVTF_DATA_TYPE::OVTF_I8:
          element_type =  element::Type_t::i8;
        case OVTF_DATA_TYPE::OVTF_U8:
          element_type =  element::Type_t::u8;
        case OVTF_DATA_TYPE::OVTF_I16:
          element_type =  element::Type_t::i16;
        case OVTF_DATA_TYPE::OVTF_U16:
          element_type =  element::Type_t::u16;
        case OVTF_DATA_TYPE::OVTF_I32:
          element_type =  element::Type_t::i32;
        case OVTF_DATA_TYPE::OVTF_I64:
          element_type =  element::Type_t::i64;
        case OVTF_DATA_TYPE::OVTF_U64:
          element_type =  element::Type_t::u64;
        case OVTF_DATA_TYPE::OVTF_BOOL:
          element_type =  element::Type_t::boolean;
        default:
          THROW_IE_EXCEPTION << "Can't convert type OVTF Data Type to nGraph Element Type!";
      }
      ngraph::Shape ng_shape(params[i].num_dims);
      for (int j=0; j<params[i].num_dims; j++)
        ng_shape[j] = params[i].dims[j];
      ie_params[i] = std::make_shared<IETensor>(element_type, ng_shape, params[i].memory_pointer);
    } else {
        ie_params[i] = nullptr;
    }
  }

  std::vector<std::shared_ptr<IETensor>> ie_outputs(num_outputs);
  std::vector<std::string> output_names(num_outputs);
  for (int i=0; i<num_outputs; i++) {
    output_names[i] = std::string(outputs[i].name);
    if (outputs[i].valid) {
      ngraph::element::Type element_type;
      switch (outputs[i].type) {
        case OVTF_DATA_TYPE::OVTF_FP32:
          element_type =  element::Type_t::f32;
        case OVTF_DATA_TYPE::OVTF_I8:
          element_type =  element::Type_t::i8;
        case OVTF_DATA_TYPE::OVTF_U8:
          element_type =  element::Type_t::u8;
        case OVTF_DATA_TYPE::OVTF_I16:
          element_type =  element::Type_t::i16;
        case OVTF_DATA_TYPE::OVTF_U16:
          element_type =  element::Type_t::u16;
        case OVTF_DATA_TYPE::OVTF_I32:
          element_type =  element::Type_t::i32;
        case OVTF_DATA_TYPE::OVTF_I64:
          element_type =  element::Type_t::i64;
        case OVTF_DATA_TYPE::OVTF_U64:
          element_type =  element::Type_t::u64;
        case OVTF_DATA_TYPE::OVTF_BOOL:
          element_type =  element::Type_t::boolean;
        default:
          THROW_IE_EXCEPTION << "Can't convert type OVTF Data Type to nGraph Element Type!";
      }
      ngraph::Shape ng_shape(outputs[i].num_dims);
      for (int j=0; j<outputs[i].num_dims; j++)
        ng_shape[j] = outputs[i].dims[j];
      ie_outputs[i] = std::make_shared<IETensor>(element_type, ng_shape, outputs[i].memory_pointer);
    } else {
        ie_outputs[i] = nullptr;
    }
  }

  if (multi_req_execution) {
    m_ie_engine->enable_multi_req_execution();
  }

  m_ie_engine->infer(ie_inputs, input_names, ie_outputs, output_names,
                     ie_params, param_names);

  for (int i=0; i<num_outputs; i++) {
    if (!(outputs[i].valid)) {
      OVTF_DATA_TYPE data_type;
      switch (ie_outputs[i]->get_blob()->getTensorDesc().getPrecision()) {
        case InferenceEngine::Precision::FP32:
          data_type = OVTF_DATA_TYPE::OVTF_FP32;
        case InferenceEngine::Precision::I8:
          data_type = OVTF_DATA_TYPE::OVTF_I8;
        case InferenceEngine::Precision::U8:
          data_type = OVTF_DATA_TYPE::OVTF_U8;
        case InferenceEngine::Precision::I16:
          data_type = OVTF_DATA_TYPE::OVTF_I16;
        case InferenceEngine::Precision::U16:
          data_type = OVTF_DATA_TYPE::OVTF_U16;
        case InferenceEngine::Precision::I32:
          data_type = OVTF_DATA_TYPE::OVTF_I32;
        case InferenceEngine::Precision::I64:
          data_type = OVTF_DATA_TYPE::OVTF_I64;
        case InferenceEngine::Precision::U64:
          data_type = OVTF_DATA_TYPE::OVTF_U64;
        case InferenceEngine::Precision::BOOL:
          data_type = OVTF_DATA_TYPE::OVTF_BOOL;
        default:
          THROW_IE_EXCEPTION << "Can't convert type OVTF Data Type to nGraph Element Type!";
      }
      ngraph::Shape ng_shape(ie_outputs[i]->get_blob()->getTensorDesc().getDims());
      outputs[i].valid = 1;
      outputs[i].type = data_type;
      outputs[i].num_dims = ng_shape.size();
      outputs[i].dims = new size_t[outputs[i].num_dims];
      for (int j=0; j<outputs[i].num_dims; j++)
        outputs[i].dims[j] = ng_shape[j];
      auto lm = InferenceEngine::as<InferenceEngine::MemoryBlob>(ie_outputs[i]->get_blob())->rwmap();
      void* out_ptr = lm.as<void*>();
      outputs[i].memory_pointer = out_ptr;
    }
  }


  return true;
}

bool ExternalExecutable::CallTrivial(const vector<shared_ptr<runtime::Tensor>>& inputs,
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
}// namespace openvino_tensorflow
}// namespace tensorflow
