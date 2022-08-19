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
