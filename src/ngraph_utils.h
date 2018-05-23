/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
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
#ifndef NGRAPH_TF_BRIDGE_UTILS_H_
#define NGRAPH_TF_BRIDGE_UTILS_H_

#include <ostream>

#include "ngraph/ngraph.hpp"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

using namespace std;
namespace tf = tensorflow;
namespace ng = ngraph;

namespace ngraph_bridge {

//
void SummarizeOp(tf::OpKernelConstruction* ctx, std::ostream& out);

// Taken from: tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc
// Extract values from a Const op to `values`. Returns true if succeeds.
template <typename T>
bool ValuesFromConstNode(const tf::NodeDef& node,
                         tf::TensorShapeProto* const_tensor_shape,
                         std::vector<T>* values) {
  if (node.op() != "Const") {
    cout << "Node Not a CONST\n";
    return false;
  }

  if (node.attr().at("dtype").type() != tf::DataTypeToEnum<T>::value) {
    cout << "Node Not a valid data type defined for CONST. Defined: "
         << node.attr().at("dtype").type() << endl;
    return false;
  }

  // TensorProto represents the content of the tensor in either <type>_val or
  // tensor_content.
  const tf::TensorProto& tensor = node.attr().at("value").tensor();
  typename tf::checkpoint::SaveTypeTraits<T>::RepeatedField* tensor_values =
      tf::checkpoint::MutableTensorProtoData<T>(
          const_cast<tf::TensorProto*>(&tensor));

  const tf::TensorShapeProto& shape = tensor.tensor_shape();
  *const_tensor_shape = shape;
  if (!tensor_values->empty() && tensor.has_tensor_shape()) {
    cout << "CONST Node has tensor shape\n";

    // When tensor_shape is set, theoretically the representation of the data
    // could be compressed. So, before copying values to the returned vector,
    // make sure no compression happens.
    if (shape.dim_size() == 1 && shape.dim(0).size() == tensor_values->size()) {
      values->insert(values->end(), tensor_values->begin(),
                     tensor_values->end());
      return true;
    }
  }

  const auto tensor_content_size = tensor.tensor_content().size();
  cout << "CONST Node tensor size: " << tensor_content_size << endl;

  if (tensor_content_size > 0) {
    CHECK_EQ(0, tensor_content_size % sizeof(T))
        << " tensor_content_size (" << tensor_content_size
        << ") is not a multiple of " << sizeof(T);
    values->resize(tensor_content_size / sizeof(T));
    tf::port::CopyToArray(tensor.tensor_content(),
                          reinterpret_cast<char*>(values->data()));
    return true;
  }

  cout << "CONST Node has empty tensor\n";
  return false;
}

// Get a scalar value from a tensor, optionally at an element offset
template <typename T>
T GetScalarFromTensorView(const std::shared_ptr<ngraph::runtime::TensorView>& t,
                          size_t element_offset = 0) {
  T result;
  t->read(&result, element_offset * sizeof(T), sizeof(T));
  return result;
}

// Prints the tensor to the given output stream
std::ostream& DumpNGTensor(
    std::ostream& s, const string& name,
    const std::shared_ptr<ngraph::runtime::TensorView>& t);

// Converts a TensorFlow DataType to an nGraph element::Type. Returns
// tf::errors::Unimplemented if the element type is not supported by nGraph
// Core. Otherwise returns Status::OK().
tf::Status TFDataTypeToNGraphElementType(tf::DataType tf_dt,
                                         ngraph::element::Type* ng_et);

}  // namespace ngraph_bridge

#endif
