/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#ifndef OPENVINO_TF_BRIDGE_BUILDER_H_
#define OPENVINO_TF_BRIDGE_BUILDER_H_

#include <ostream>
#include <vector>
#include <mutex>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"

#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/tensorflow/frontend.hpp"

#include "openvino_tensorflow/ovtf_graph_iterator.h"

namespace tensorflow {
namespace openvino_tensorflow {

class Builder {
 public:
  static Status TranslateGraph(
      const std::vector<TensorShape>& inputs,
      const std::vector<const Tensor*>& static_input_map, const Graph* tf_graph,
      const string name, std::shared_ptr<ov::Model>& ng_function);

  static Status TranslateGraph(
      const std::vector<TensorShape>& inputs,
      const std::vector<const Tensor*>& static_input_map, const Graph* tf_graph,
      const string name, std::shared_ptr<ov::Model>& ng_function,
      ov::ResultVector& ng_func_result_list,
      const std::vector<Tensor>& tf_input_tensors);

  static Status TranslateGraphWithTFFE(
      const std::vector<TensorShape>& inputs,
      const std::vector<const Tensor*>& static_input_map,
      const Graph* input_graph, const string name,
      std::shared_ptr<ngraph::Function>& ng_function,
      ngraph::ResultVector& zero_dim_outputs,
      const std::vector<Tensor>& tf_input_tensors);

  using OpMap =
      std::unordered_map<std::string, std::vector<ov::Output<ov::Node>>>;
  using ConstMap =
      std::map<DataType,
               std::pair<std::function<Status(const Node*, ov::element::Type,
                                              ov::Output<ov::Node>&)>,
                         const ov::element::Type>>;
  static const Builder::ConstMap& TF_NGRAPH_CONST_MAP();

  template <typename T>
  static void MakePadding(const std::string& tf_padding_type,
                          const ov::Shape& ng_image_shape,
                          const ov::Shape& ng_kernel_shape,
                          const ov::Strides& ng_strides,
                          const ov::Shape& ng_dilations, T& ng_padding_below,
                          T& ng_padding_above) {
    if (tf_padding_type == "SAME") {
      ov::Shape img_shape = {0, 0};
      img_shape.insert(img_shape.end(), ng_image_shape.begin(),
                       ng_image_shape.end());
      ov::infer_auto_padding(img_shape, ng_kernel_shape, ng_strides,
                             ng_dilations, ov::op::PadType::SAME_UPPER,
                             ng_padding_above, ng_padding_below);
    } else if (tf_padding_type == "VALID") {
      ng_padding_below.assign(ng_image_shape.size(), 0);
      ng_padding_above.assign(ng_image_shape.size(), 0);
    }
  }

  // This function is used to trace which ng node came from which tf node
  // It does 3 things:
  // 1. Attaches provenance tags. This is guaranteed to propagate the tag info
  // to all nodes.
  // The next 2 are not guaranteed to be present for all nodes.
  // But when present they are correct and agree with provenance tags
  // 2. Attaches friendly names.
  // 3. Prints a log if OPENVINO_TF_LOG_PLACEMENT=1
  static void SetTracingInfo(const std::string& op_name,
                             const ov::Output<ov::Node> ng_node);

  static void SetLibPath(const std::string&);

  template <typename T>
  static void values_from_tensorproto(
      const ::tensorflow::TensorProto tensor_proto,
      const ngraph::element::Type dt, ov::Shape* const_tensor_shape,
      std::vector<T>* values) {
    const ::tensorflow::TensorShapeProto& tf_shape =
        tensor_proto.tensor_shape();
    std::vector<ov::Dimension> dims;
    for (int i = 0; i < tf_shape.dim_size(); i++) {
      dims.emplace_back(tf_shape.dim(i).size());
    }
    ov::PartialShape pshape(dims);
    if (!(pshape.is_static())) {
      throw errors::InvalidArgument(
          "Dynamic constant input shapes are not supported in _Arg "
          "conversion.");
    }
    *const_tensor_shape = pshape.get_shape();
    auto tensor_content = tensor_proto.tensor_content();
    std::vector<char> tensor_values_plain(tensor_content.begin(),
                                          tensor_content.end());
    const T* tensor_values =
        reinterpret_cast<const T*>(tensor_values_plain.data());

    if (!tensor_values_plain.empty() && tensor_proto.has_tensor_shape()) {
      // When tensor_shape is set, theoretically the representation of the data
      // could be compressed. So, before copying values to the returned vector,
      // make sure no compression happens.
      // if (shape.dim_size() == 1 && shape.dim(0).size() ==
      // tensor_values_plain.size()/sizeof(T)) {
      values->insert(values->end(), tensor_values,
                     tensor_values + tensor_values_plain.size() / sizeof(T));
      return;
      //}
    }
    const auto tensor_content_size = tensor_proto.tensor_content().size();
    if (tensor_content_size % sizeof(T)) {
      std::cerr << "[ ERROR ] tensor_content_size (" << tensor_content_size
                << ") is not a multiple of " << sizeof(T);
    }

    // If tensor_content_size is zero, we'll have to take the values from
    // int_val, float_val, etc.
    if (tensor_content_size == 0) {
      int64_t n_elements = 1;
      for (auto i = 0; i < tf_shape.dim_size(); i++) {
        if (tf_shape.dim(i).size() < 0) {
          THROW_IE_EXCEPTION
              << "Const node has empty tensor and an unknown dimension size";
        }
        n_elements *= tf_shape.dim(i).size();
      }
      values->resize(n_elements);

      auto val_lastsaved = (T)0;  // cast
      for (auto i = 0; i < n_elements; i++) {
        int64_t val_size = 0;
        auto val_i = (T)0;  // cast
        switch (dt) {
          // TODO: there are more element types to support
          // here
          case ngraph::element::Type_t::i32:
            val_size = tensor_proto.int_val_size();
            if (val_size > 0) val_i = tensor_proto.int_val()[i];
            break;
          case ngraph::element::Type_t::i64:
            val_size = tensor_proto.int64_val_size();
            if (val_size > 0) val_i = tensor_proto.int64_val()[i];
            break;
          case ngraph::element::Type_t::f32:
            val_size = tensor_proto.float_val_size();
            if (val_size > 0) val_i = tensor_proto.float_val()[i];
            break;
          case ngraph::element::Type_t::boolean:
            val_size = tensor_proto.bool_val_size();
            if (val_size > 0) val_i = tensor_proto.bool_val()[i];
            break;
          case ngraph::element::Type_t::f64:
            val_size = tensor_proto.double_val_size();
            if (val_size > 0) val_i = tensor_proto.double_val()[i];
            break;
          default:
            FRONT_END_THROW(
                "Encountered unknown element type on an empty tensor_proto");
        }
        if (val_size == 0) {
          (*values)[i] = static_cast<T>(0);
        } else if (i < val_size) {
          (*values)[i] = val_i;
          val_lastsaved = val_i;
        } else {
          (*values)[i] = val_lastsaved;
        }
      }
    } else {
      return;
    }
  }

 private:
  // tf_conversion_extensions module lib path, to load the library using
  // Frontend
  static std::string m_tf_conversion_extensions_lib_path;
  static ov::frontend::FrontEnd::Ptr m_frontend_ptr;
  static std::mutex m_translate_lock_;
};

}  // namespace openvino_tensorflow
}  // namespace tensorflow

#endif  // OPENVINO_TF_BRIDGE_BUILDER_H_
