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
#include "ngraph_utils.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/protobuf.h"

using namespace std;

namespace ngraph_bridge {
extern const char* const DEVICE_NGRAPH;

void SummarizeOp(tf::OpKernelConstruction* ctx, std::ostream& out) {
  auto node_def = ctx->def();
  out << "Node name: " << node_def.name() << " Op: " << node_def.op() << "\n";
  out << "Inputs: " << node_def.input().size() << "\n    ";
  for (const std::string& input : node_def.input()) {
    out << input << "\n    ";
  }
  out << "\n";
}

std::ostream& DumpNGTensor(
    std::ostream& s, const string& name,
    const std::shared_ptr<ngraph::runtime::TensorView>& t) {
  // std::shared_ptr<ngraph::runtime::TensorView> t{get_tensor()};
  const ngraph::Shape& shape = t->get_shape();
  s << "Tensor<" << name << ": ";

  for (size_t i = 0; i < shape.size(); ++i) {
    s << shape.at(i);
    if (i + 1 < shape.size()) {
      s << ", ";
    }
  }
  size_t pos = 0;
  s << ">{";
  size_t rank = shape.size();
  if (rank == 0) {
    s << GetScalarFromTensorView<float>(t, pos++);
  } else if (rank <= 2) {
    s << "[";
    for (size_t i = 0; i < shape.at(0); ++i) {
      if (rank == 1) {
        s << GetScalarFromTensorView<float>(t, pos++);
      } else if (rank == 2) {
        s << "[";
        for (size_t j = 0; j < shape.at(1); ++j) {
          s << GetScalarFromTensorView<float>(t, pos++);

          if (j + 1 < shape.at(1)) {
            s << ", ";
          }
        }
        s << "]";
      }
      if (i + 1 < shape.at(0)) {
        s << ", ";
      }
    }
    s << "]";
  }
  s << "}";
  return s;
}

tf::Status TFDataTypeToNGraphElementType(tf::DataType tf_dt,
                                         ngraph::element::Type* ng_et) {
  switch (tf_dt) {
    case tf::DataType::DT_FLOAT:
      *ng_et = ng::element::f32;
      break;
    case tf::DataType::DT_DOUBLE:
      *ng_et = ng::element::f64;
      break;
    case tf::DataType::DT_INT32:
      *ng_et = ng::element::i32;
      break;
    case tf::DataType::DT_UINT8:
      *ng_et = ng::element::u8;
      break;
    case tf::DataType::DT_INT64:
      *ng_et = ng::element::i64;
      break;
    case tf::DataType::DT_UINT32:
      *ng_et = ng::element::u32;
      break;
    case tf::DataType::DT_UINT64:
      *ng_et = ng::element::u64;
      break;
    case tf::DataType::DT_BOOL:
      *ng_et = ng::element::boolean;
      break;
    default:
      return tf::errors::Unimplemented("Unsupported TensorFlow data type: ",
                                       tf::DataType_Name(tf_dt));
  }

  return tf::Status::OK();
}

tf::Status TFTensorShapeToNGraphShape(const tf::TensorShape& tf_shape,
                                      ngraph::Shape* ng_shape) {
  for (int i = 0; i < tf_shape.dims(); i++) {
    if (tf_shape.dim_size(i) < 0) {
      return tf::errors::InvalidArgument(
          "TensorFlow shape has a negative dimension size");
    }
  }

  *ng_shape = ngraph::Shape(tf_shape.dims());
  for (int i = 0; i < tf_shape.dims(); i++) {
    (*ng_shape)[i] = tf_shape.dim_size(i);
  }

  return tf::Status::OK();
}

const tf::gtl::ArraySlice<tf::DataType>& NGraphDTypes() {
  static tf::gtl::ArraySlice<tf::DataType> result{
      tf::DT_FLOAT,  tf::DT_DOUBLE, tf::DT_INT8,  tf::DT_INT16,
      tf::DT_INT32,  tf::DT_INT64,  tf::DT_UINT8, tf::DT_UINT16,
      tf::DT_UINT32, tf::DT_UINT64, tf::DT_BOOL};
  return result;
}

const tf::gtl::ArraySlice<tf::DataType>& NGraphNumericDTypes() {
  static tf::gtl::ArraySlice<tf::DataType> result{
      tf::DT_FLOAT, tf::DT_DOUBLE, tf::DT_INT8,   tf::DT_INT16,  tf::DT_INT32,
      tf::DT_INT64, tf::DT_UINT8,  tf::DT_UINT16, tf::DT_UINT32, tf::DT_UINT64};
  return result;
}

const tf::gtl::ArraySlice<tf::DataType>& NGraphIndexDTypes() {
  static tf::gtl::ArraySlice<tf::DataType> result{tf::DT_INT32, tf::DT_INT64};
  return result;
}

}  // namespace ngraph_bridge
