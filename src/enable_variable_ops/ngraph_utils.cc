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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/protobuf.h"

#include "ngraph_utils.h"
using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

// static int testing_graph_id=30;

Status IsCopyLogEnabled(int graph_id, bool& is_copy_log_enabled) {
  const char* copy_env_var = std::getenv("NGRAPH_TF_LOG_COPIES");
  if (copy_env_var == nullptr) {
    is_copy_log_enabled = false;
    return Status::OK();
  }
  int test_graph_id;

  try {
    test_graph_id = stoi(string(copy_env_var));
  } catch (const std::invalid_argument& ia) {
    return errors::InvalidArgument("Invalid argument for NGRAPH_TF_LOG_COPIES");
  }

  // if -1 copies are logged for all graphs
  is_copy_log_enabled = (test_graph_id == -1 || test_graph_id == graph_id);
  return Status::OK();
}

void PrintTFTensor(Tensor& T1) {
  NGRAPH_VLOG(4) << "all tensor values" << (T1).SummarizeValue(64) << endl;
}

std::string DebugNode(Node* node) {
  std::string temp = node->name();
  temp += "[" + node->type_string() + "]";
  return temp;
}

std::string PrintBool(bool var) { return (var ? "Yes" : "No"); }

bool IsNGVariableType(string node_type) {
  return (node_type == "NGraphVariable" || node_type == "NGraphAssign");
};

bool IsNGSupportedType(string node_type) {
  return (IsNGVariableType(node_type) || node_type == "NGraphEncapsulate");
};

// Read from this ng_tensor into tf_tensor
void ReadNGTensor(shared_ptr<ng::runtime::Tensor> ng_tensor,
                  Tensor* tf_tensor) {
  void* tf_src_ptr = (void*)DMAHelper::base(tf_tensor);
  ng_tensor->read(tf_src_ptr, 0, ng_tensor->get_element_count() *
                                     ng_tensor->get_element_type().size());
}

// Write into this ng_tensor from tf_tensor
void WriteNGTensor(shared_ptr<ng::runtime::Tensor> ng_tensor,
                   Tensor* tf_tensor) {
  void* tf_src_ptr = (void*)DMAHelper::base(tf_tensor);
  ng_tensor->write(tf_src_ptr, 0, ng_tensor->get_element_count() *
                                      ng_tensor->get_element_type().size());
}

void SummarizeOp(OpKernelConstruction* ctx, std::ostream& out) {
  auto node_def = ctx->def();
  out << "Node name: " << node_def.name() << " Op: " << node_def.op() << "\n";
  out << "Inputs: " << node_def.input().size() << "\n    ";
  for (const std::string& input : node_def.input()) {
    out << input << "\n    ";
  }
  out << "\n";
}

std::ostream& DumpNGTensor(std::ostream& s, const string& name,
                           const std::shared_ptr<ngraph::runtime::Tensor>& t) {
  // std::shared_ptr<ngraph::runtime::Tensor> t{get_tensor()};
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
    s << GetScalarFromTensor<float>(t, pos++);
  } else if (rank <= 2) {
    s << "[";
    for (size_t i = 0; i < shape.at(0); ++i) {
      if (rank == 1) {
        s << GetScalarFromTensor<float>(t, pos++);
      } else if (rank == 2) {
        s << "[";
        for (size_t j = 0; j < shape.at(1); ++j) {
          s << GetScalarFromTensor<float>(t, pos++);

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

Status TFDataTypeToNGraphElementType(DataType tf_dt,
                                     ngraph::element::Type* ng_et) {
  switch (tf_dt) {
    case DataType::DT_FLOAT:
      *ng_et = ng::element::f32;
      break;
    case DataType::DT_DOUBLE:
      *ng_et = ng::element::f64;
      break;
    case DataType::DT_INT32:
      *ng_et = ng::element::i32;
      break;
    case DataType::DT_UINT8:
      *ng_et = ng::element::u8;
      break;
    case DataType::DT_UINT16:
      *ng_et = ng::element::u16;
      break;
    case DataType::DT_INT64:
      *ng_et = ng::element::i64;
      break;
    case DataType::DT_UINT32:
      *ng_et = ng::element::u32;
      break;
    case DataType::DT_UINT64:
      *ng_et = ng::element::u64;
      break;
    case DataType::DT_BOOL:
      *ng_et = ng::element::boolean;
      break;
    case DataType::DT_QINT8:
      *ng_et = ng::element::i8;
      break;
    case DataType::DT_QUINT8:
      *ng_et = ng::element::u8;
      break;
    case DataType::DT_QINT32:
      *ng_et = ng::element::i32;
      break;
    default:
      return errors::Unimplemented("Unsupported TensorFlow data type: ",
                                   DataType_Name(tf_dt));
  }
  return Status::OK();
}

Status TFTensorShapeToNGraphShape(const TensorShape& tf_shape,
                                  ngraph::Shape* ng_shape) {
  for (int i = 0; i < tf_shape.dims(); i++) {
    if (tf_shape.dim_size(i) < 0) {
      return errors::InvalidArgument(
          "TensorFlow shape has a negative dimension size");
    }
  }

  *ng_shape = ngraph::Shape(tf_shape.dims());
  for (int i = 0; i < tf_shape.dims(); i++) {
    (*ng_shape)[i] = tf_shape.dim_size(i);
  }

  return Status::OK();
}

void print_node_histogram(const std::unordered_map<string, int>& histogram,
                          bool sorted) {
  int histogram_size = histogram.size();
  if (histogram_size == 0) {
    std::cout << "None";
  } else {
    vector<std::pair<string, int>> vec(begin(histogram), end(histogram));
    if (sorted) {
      sort(begin(vec), end(vec),
           [](const pair<string, int>& a, const pair<string, int>& b) {
             // descending sort
             return a.second > b.second;
           });
    }

    for (auto node : vec) {
      bool endelem = node == vec.back();
      std::cout << " " << node.first << " -> " << node.second
                << (endelem ? " " : ",");
    }
  }
}

const gtl::ArraySlice<DataType>& NGraphDTypes() {
  static gtl::ArraySlice<DataType> result{
      DT_FLOAT,  DT_DOUBLE, DT_INT8,   DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
      DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL,  DT_QINT8, DT_QUINT8};
  return result;
}

const gtl::ArraySlice<DataType>& NGraphNumericDTypes() {
  static gtl::ArraySlice<DataType> result{
      DT_FLOAT, DT_DOUBLE, DT_INT8,   DT_INT16,  DT_INT32,
      DT_INT64, DT_UINT8,  DT_UINT16, DT_UINT32, DT_UINT64};
  return result;
}

const gtl::ArraySlice<DataType>& NGraphNumericAndQuantizedDTypes() {
  static gtl::ArraySlice<DataType> result{
      DT_FLOAT, DT_DOUBLE, DT_INT8,   DT_INT16,  DT_INT32, DT_INT64,
      DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_QINT8, DT_QUINT8};
  return result;
}

const gtl::ArraySlice<DataType>& NGraphIndexDTypes() {
  static gtl::ArraySlice<DataType> result{DT_INT32, DT_INT64};
  return result;
}

const gtl::ArraySlice<DataType>& NGraphSupportedQuantizedDTypes() {
  static gtl::ArraySlice<DataType> result{DT_QINT8, DT_QUINT8};
  return result;
}

const gtl::ArraySlice<DataType>& NGraphRealDTypes() {
  static gtl::ArraySlice<DataType> result{DT_FLOAT, DT_DOUBLE};
  return result;
}

const gtl::ArraySlice<DataType>& NGraphBiasDTypes() {
  static gtl::ArraySlice<DataType> result{DT_FLOAT, DT_QINT32};
  return result;
}

Status CheckAxisDimInRange(std::vector<int64> axes, size_t rank) {
  for (auto i : axes) {
    if (i < (int)-rank || i >= (int)rank) {
      return errors::InvalidArgument("Axis Dimension is out of range. Got ", i,
                                     ", should be in range [-", rank, ", ",
                                     rank, ")");
    }
  }
  return Status::OK();
}

void NgraphSerialize(const std::string& file_name,
                     const std::shared_ptr<ngraph::Function>& ng_function) {
  NGRAPH_VLOG(0) << "Serializing graph to: " << file_name << std::endl;
  std::string js = ngraph::serialize(ng_function, 4);
  std::ofstream f;
  f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  try {
    f.open(file_name);
    f << js;
    f.close();
  } catch (std::ofstream::failure& e) {
    NGRAPH_VLOG(0) << "Exception opening/closing file " << file_name
                   << std::endl;
    NGRAPH_VLOG(0) << e.what() << std::endl;
  }
}

void MemoryProfile(long& vm_usage, long& resident_set) {
  vm_usage = 0;
  resident_set = 0;

  // Get the two fields we want
  long vsize;
  long rss;

  std::ifstream ifs("/proc/self/stat", std::ios_base::in);
  std::string mem_in;
  getline(ifs, mem_in);
  if (mem_in != "") {
    vector<string> mem_str = ng::split(mem_in, ' ');
    vsize = std::stol(mem_str[22]);
    rss = std::stol(mem_str[23]);

    long page_size_kb = sysconf(_SC_PAGE_SIZE) /
                        1024;  // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024;   // unit kb
    resident_set = rss * page_size_kb;
  }
};

}  // namespace ngraph_bridge

}  // namespace tensorflow
