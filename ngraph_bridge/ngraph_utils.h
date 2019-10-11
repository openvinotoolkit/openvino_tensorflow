/*******************************************************************************
 * Copyright 2017-2019 Intel Corporation
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

#include <fstream>
#include <ostream>
#include <sstream>

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

#include "ngraph/event_tracing.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/serializer.hpp"

#include "logging/ngraph_log.h"

namespace ng = ngraph;
using namespace std;
namespace tensorflow {

namespace ngraph_bridge {

/* -------------------------------------------------
//
// NGraphVariableMap : Map of Variable names and their backend tensors
//
---------------------------------------------------*/

Status IsNgraphTFLogTensorCopiesEnabled(int graph_id,
                                        bool& is_copy_log_enabled);

Status GetNgraphVarBufferSharingState(int& buffer_sharing_state);

void PrintTFTensor(Tensor& T1);
std::string DebugNode(Node* node);

// Read from this ng_tensor into tf_tensor
void ReadNGTensor(shared_ptr<ng::runtime::Tensor> ng_tensor, Tensor* tf_tensor);
std::string PrintBool(bool var);

// Write into this ng_tensor from tf_tensor
void WriteNGTensor(shared_ptr<ng::runtime::Tensor> ng_tensor,
                   Tensor* tf_tensor);

void SummarizeOp(OpKernelConstruction* ctx, std::ostream& out);

// Node-types on a variable and are executed on nGraph
bool IsNGVariableType(string node_type);

// Node-types that are executed on nGraph
bool IsNGSupportedType(string node_type);

bool IsProcessedByNgraphPass(Graph* g);

// Taken from: tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc
// Extract values from a Const op to `values`. Returns true if succeeds.
//
// Modified with an extra `VecT` parameter to handle the case where the type
// in the vector does not match TensorFlow's notion of what the C++ type
// should be (e.g. when T is `bool`, we actually need a vector of `char` for
// compatibility with nGraph).
template <typename T, typename VecT = T>
Status ValuesFromConstNode(const NodeDef& node,
                           TensorShapeProto* const_tensor_shape,
                           std::vector<VecT>* values) {
  if (node.op() != "Const") {
    return errors::InvalidArgument("Node not a Const");
  }

  if (node.attr().at("dtype").type() != DataTypeToEnum<T>::value) {
    std::stringstream ss;
    ss << "Invalid data type defined for Const. Defined: "
       << node.attr().at("dtype").type();
    return errors::InvalidArgument(ss.str());
  }

  // TensorProto represents the content of the tensor in either <type>_val or
  // tensor_content.
  const TensorProto& tensor = node.attr().at("value").tensor();
  typename checkpoint::SaveTypeTraits<T>::RepeatedField* tensor_values =
      checkpoint::MutableTensorProtoData<T>(const_cast<TensorProto*>(&tensor));

  const TensorShapeProto& shape = tensor.tensor_shape();
  *const_tensor_shape = shape;
  if (!tensor_values->empty() && tensor.has_tensor_shape()) {
    // When tensor_shape is set, theoretically the representation of the data
    // could be compressed. So, before copying values to the returned vector,
    // make sure no compression happens.
    if (shape.dim_size() == 1 && shape.dim(0).size() == tensor_values->size()) {
      values->insert(values->end(), tensor_values->begin(),
                     tensor_values->end());
      return Status::OK();
    }
  }

  const auto tensor_content_size = tensor.tensor_content().size();
  CHECK_EQ(0, tensor_content_size % sizeof(VecT))
      << " tensor_content_size (" << tensor_content_size
      << ") is not a multiple of " << sizeof(VecT);

  // If tensor_content_size is zero, we'll have to take the values from
  // int_val, float_val, etc.
  if (tensor_content_size == 0) {
    int64 n_elements = 1;
    for (auto i = 0; i < shape.dim_size(); i++) {
      if (shape.dim(i).size() < 0) {
        return errors::InvalidArgument(
            "Const node has empty tensor and an unknown dimension size");
      }
      n_elements *= shape.dim(i).size();
    }
    values->resize(n_elements);
    for (auto i = 0; i < n_elements; i++) {
      auto& tensor = node.attr().at("value").tensor();
      auto dt = node.attr().at("dtype").type();
      switch (dt) {
        // TODO(amprocte/NGRAPH-2502): there are more element types to support
        // here
        case DT_INT32:
          (*values)[i] = (tensor.int_val_size() == 1 ? tensor.int_val()[0]
                                                     : tensor.int_val()[i]);
          break;
        case DT_INT64:
          (*values)[i] = (tensor.int64_val_size() == 1 ? tensor.int64_val()[0]
                                                       : tensor.int64_val()[i]);
          break;
        case DT_FLOAT:
          (*values)[i] = (tensor.float_val_size() == 1 ? tensor.float_val()[0]
                                                       : tensor.float_val()[i]);
          break;
        case DT_BOOL:
          (*values)[i] = (tensor.bool_val_size() == 1 ? tensor.bool_val()[0]
                                                      : tensor.bool_val()[i]);
          break;
        case DT_DOUBLE:
          (*values)[i] =
              (tensor.double_val_size() == 1 ? tensor.double_val()[0]
                                             : tensor.double_val()[i]);
          break;
        default:
          NGRAPH_VLOG(0)
              << "Const node has empty tensor and we don't know how to "
                 "handle this element type";
          NGRAPH_VLOG(0) << node.DebugString();
          NGRAPH_VLOG(0) << shape.DebugString();
          return errors::Unimplemented("Encountered unknown element type ",
                                       DataType_Name(dt),
                                       " on an empty tensor");
      }
    }
  } else {
    values->resize(tensor_content_size / sizeof(VecT));
    port::CopyToArray(tensor.tensor_content(),
                      reinterpret_cast<char*>(values->data()));
  }

  return Status::OK();
}

// Get a scalar value from a tensor, optionally at an element offset
template <typename T>
T GetScalarFromTensor(const std::shared_ptr<ngraph::runtime::Tensor>& t,
                      size_t element_offset = 0) {
  T result;
  t->read(&result, element_offset * sizeof(T), sizeof(T));
  return result;
}

// // Descending sort the map based on the value
void print_node_histogram(const std::unordered_map<string, int>&,
                          bool sorted = true);

// Prints the tensor to the given output stream
// TODO: internally convert ng types to cpptypes
// so that users do not have to specify the template arg T
template <typename T>
std::ostream& DumpNGTensor(std::ostream& s, const string& name,
                           const std::shared_ptr<ngraph::runtime::Tensor>& t) {
  // std::shared_ptr<ngraph::runtime::Tensor> t{get_tensor()};
  const ngraph::Shape& shape = t->get_shape();
  s << "Tensor<" << name << ": ";
  auto type = t->get_element_type();
  bool T_is_integral = std::is_integral<T>::value;
  bool type_is_integral = type.is_integral();
  if (type_is_integral != T_is_integral) {
    std::stringstream err_msg;
    err_msg << "Tensor type " << type << " is"
            << (type_is_integral ? " " : " not ")
            << "integral but passed template is"
            << (T_is_integral ? " " : " not ") << "integral";
    throw std::invalid_argument(err_msg.str());
  }

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
    s << GetScalarFromTensor<T>(t, pos++);
  } else if (rank <= 2) {
    s << "[";
    for (size_t i = 0; i < shape.at(0); ++i) {
      if (rank == 1) {
        s << GetScalarFromTensor<T>(t, pos++);
      } else if (rank == 2) {
        s << "[";
        for (size_t j = 0; j < shape.at(1); ++j) {
          s << GetScalarFromTensor<T>(t, pos++);

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
  // TODO: extend for > 2 rank
  s << "}";
  return s;
}

template <typename T>
static void TensorDataToStream(std::ostream& ostream, int64 n_elements,
                               const char* data) {
  const T* data_T = reinterpret_cast<const T*>(data);
  for (size_t i = 0; i < n_elements; i++) {
    ostream << data_T[i] << ",";
  }
}

Status TensorToStream(std::ostream& ostream, const Tensor& tensor);

// Converts a TensorFlow DataType to an nGraph element::Type. Returns
// errors::Unimplemented if the element type is not supported by nGraph
// Core. Otherwise returns Status::OK().
Status TFDataTypeToNGraphElementType(DataType tf_dt,
                                     ngraph::element::Type* ng_et);

// Converts a TensorFlow TensorShape to an nGraph Shape. Requires that none of
// the dimension lengths in tf_shape are negative.
Status TFTensorShapeToNGraphShape(const TensorShape& tf_shape,
                                  ngraph::Shape* ng_shape);

// Returns an ArraySlice containing all TensorFlow dtypes supported by the
// nGraph bridge.
const gtl::ArraySlice<DataType>& NGraphDTypes();

// Returns an ArraySlice containing all *numeric* TensorFlow dtypes supported
// by the nGraph bridge.
const gtl::ArraySlice<DataType>& NGraphNumericDTypes();

// Returns an ArraySlice containing all numeric and quantized TensorFlow dtypes
// supported by the nGraph bridge.
const gtl::ArraySlice<DataType>& NGraphNumericAndQuantizedDTypes();

// Returns an ArraySlice containing all data types that can be used for
// axis/tensor indices.
const gtl::ArraySlice<DataType>& NGraphIndexDTypes();

// Returns an ArraySlice containing supported data types in the quantized domain
const gtl::ArraySlice<DataType>& NGraphSupportedQuantizedDTypes();

// Returns an ArraySlice containing supported real/non-integer data types
const gtl::ArraySlice<DataType>& NGraphRealDTypes();

// Returns an ArraySlice containing supported bias types for custom quant op
const gtl::ArraySlice<DataType>& NGraphBiasDTypes();

// Check to make sure the axis dimension for reduction are in within range.
// Returns error if axis is out of range. Otherwise returns Status::OK().
Status CheckAxisDimInRange(std::vector<int64> axes, size_t rank);

// Serialize a ngraph function into a file
Status NgraphSerialize(const std::string&,
                       const std::shared_ptr<ngraph::Function>&);

// Dump given string to file
Status StringToFile(const std::string&, const std::string&,
                    bool sanitize_name = true);

// Remove '/' from file name (which might appear due to say, tf scopes)
string SanitizeFileName(const string file_name);

// Collect the total memory usage through /proc/self/stat
void MemoryProfile(long&, long&);

std::string DotFilename(std::string, int);

std::string DotFilename(std::string kind, int idx, int sub_idx);

std::string PbtxtFilename(std::string, int);

std::string PbtxtFilename(std::string kind, int idx, int sub_idx);

std::string GraphFilenamePrefix(std::string, int);

std::string GraphFilenamePrefix(std::string, int, int);

bool DumpAllGraphs();

bool DumpPrecaptureGraphs();

bool DumpCapturedGraphs();

bool DumpUnmarkedGraphs();

bool DumpMarkedGraphs();

bool DumpClusteredGraphs();

bool DumpDeclusteredGraphs();

bool DumpEncapsulatedGraphs();

bool DumpTrackedGraphs();

#if defined(NGRAPH_DISTRIBUTED)
// Insert constrol dependency for AllReduce ops to ensure execution order
void OpControlOrder(const std::shared_ptr<ngraph::Function>&,
                    const std::string&);
#endif

}  // namespace ngraph_bridge

}  // namespace tensorflow

#endif
