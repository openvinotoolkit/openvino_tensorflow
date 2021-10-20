/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#ifndef OPENVINO_TF_BRIDGE_BUILDER_H_
#define OPENVINO_TF_BRIDGE_BUILDER_H_

#include <ostream>
#include <vector>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"

#include "ngraph/ngraph.hpp"

namespace tensorflow {
namespace openvino_tensorflow {

class Builder {
 public:
  static Status TranslateGraph(
      const std::vector<TensorShape>& inputs,
      const std::vector<const Tensor*>& static_input_map, const Graph* tf_graph,
      const string name, std::shared_ptr<ngraph::Function>& ng_function, const std::vector<Tensor>& tf_input_tensors);

  static Status TranslateGraph(
      const std::vector<TensorShape>& inputs,
      const std::vector<const Tensor*>& static_input_map, const Graph* tf_graph,
      const string name, std::shared_ptr<ngraph::Function>& ng_function,
      ngraph::ResultVector& ng_func_result_list, const std::vector<Tensor>& tf_input_tensors);

  using OpMap = std::unordered_map<std::string,
                                   std::vector<ngraph::Output<ngraph::Node>>>;
  using ConstMap = std::map<
      DataType,
      std::pair<std::function<Status(const Node*, ngraph::element::Type,
                                     ngraph::Output<ngraph::Node>&)>,
                const ngraph::element::Type>>;
  static const Builder::ConstMap& TF_NGRAPH_CONST_MAP();

  template <typename T>
  static void MakePadding(const std::string& tf_padding_type,
                          const ngraph::Shape& ng_image_shape,
                          const ngraph::Shape& ng_kernel_shape,
                          const ngraph::Strides& ng_strides,
                          const ngraph::Shape& ng_dilations,
                          T& ng_padding_below, T& ng_padding_above) {
    if (tf_padding_type == "SAME") {
      ngraph::Shape img_shape = {0, 0};
      img_shape.insert(img_shape.end(), ng_image_shape.begin(),
                       ng_image_shape.end());
      ngraph::infer_auto_padding(img_shape, ng_kernel_shape, ng_strides,
                                 ng_dilations, ngraph::op::PadType::SAME_UPPER,
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
                             const ngraph::Output<ngraph::Node> ng_node);
};

}  // namespace openvino_tensorflow
}  // namespace tensorflow

#endif  // OPENVINO_TF_BRIDGE_BUILDER_H_
