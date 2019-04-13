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

#if defined NGRAPH_DISTRIBUTED
#include "ngraph/distributed.hpp"
#endif

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/protobuf.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

bool IsNGVariableType(string node_type) {
  return node_type == "NGraphVariable";
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
}

std::string DotFilename(std::string kind, int idx) {
  return GraphFilenamePrefix(kind, idx) + ".dot";
}

std::string DotFilename(std::string kind, int idx, int sub_idx) {
  return GraphFilenamePrefix(kind, idx, sub_idx) + ".dot";
}

std::string PbtxtFilename(std::string kind, int idx) {
  return GraphFilenamePrefix(kind, idx) + ".pbtxt";
}

std::string PbtxtFilename(std::string kind, int idx, int sub_idx) {
  return GraphFilenamePrefix(kind, idx, sub_idx) + ".pbtxt";
}

std::string GraphFilenamePrefix(std::string kind, int idx) {
  std::stringstream ss;
  ss << kind << "_" << std::setfill('0') << std::setw(4) << idx;
#if defined NGRAPH_DISTRIBUTED
  ngraph::Distributed dist;
  int Rank_ID = dist.get_rank();
  ss << "_" << std::setfill('0') << std::setw(4) << Rank_ID;
#endif
  return ss.str();
}

std::string GraphFilenamePrefix(std::string kind, int idx, int sub_idx) {
  std::stringstream ss;
  ss << GraphFilenamePrefix(kind, idx) << "_" << std::setfill('0')
     << std::setw(4) << sub_idx;
#if defined NGRAPH_DISTRIBUTED
  ngraph::Distributed dist;
  int Rank_ID = dist.get_rank();
  ss << "_" << std::setfill('0') << std::setw(4) << Rank_ID;
#endif
  return ss.str();
}

bool DumpAllGraphs() { return std::getenv("NGRAPH_TF_DUMP_GRAPHS") != nullptr; }

bool DumpPrecaptureGraphs() {
  return DumpAllGraphs() ||
         std::getenv("NGRAPH_TF_DUMP_PRE_CAPTURED_GRAPHS") != nullptr;
}

bool DumpCapturedGraphs() {
  return DumpAllGraphs() ||
         std::getenv("NGRAPH_TF_DUMP_CAPTURED_GRAPHS") != nullptr;
}

bool DumpUnmarkedGraphs() {
  return DumpAllGraphs() ||
         std::getenv("NGRAPH_TF_DUMP_UNMARKED_GRAPHS") != nullptr;
}

bool DumpMarkedGraphs() {
  return DumpAllGraphs() ||
         std::getenv("NGRAPH_TF_DUMP_MARKED_GRAPHS") != nullptr;
}

bool DumpClusteredGraphs() {
  return DumpAllGraphs() ||
         std::getenv("NGRAPH_TF_DUMP_CLUSTERED_GRAPHS") != nullptr;
}

bool DumpDeclusteredGraphs() {
  return DumpAllGraphs() ||
         std::getenv("NGRAPH_TF_DUMP_DECLUSTERED_GRAPHS") != nullptr;
}

bool DumpEncapsulatedGraphs() {
  return DumpAllGraphs() ||
         std::getenv("NGRAPH_TF_DUMP_ENCAPSULATED_GRAPHS") != nullptr;
}

bool DumpTrackedGraphs() {
  return DumpAllGraphs() ||
         std::getenv("NGRAPH_TF_DUMP_TRACKED_GRAPHS") != nullptr;
}

void AllreduceOpControlOrder(
    const std::shared_ptr<ngraph::Function>& ng_function) {
  // Get the serialized ops and stored the allreduce ops to a vector and
  ng::NodeVector allreduce_op_list;
  for (const shared_ptr<ng::Node>& node : ng_function->get_ordered_ops()) {
    if (node->description() == "AllReduce") {
      allreduce_op_list.push_back(node);
    }
    // Sort the allreduce ops according to the TF names
    std::sort(allreduce_op_list.begin(), allreduce_op_list.end(),
              [](const shared_ptr<ng::Node>& x, const shared_ptr<ng::Node>& y) {
                return x->get_friendly_name() < y->get_friendly_name();
              });
    // Add control dependency in for the allreduce ops
    if (allreduce_op_list.size() > 1) {
      for (size_t i = 1; i < allreduce_op_list.size(); ++i) {
        auto pre_node = allreduce_op_list[i - 1];
        auto cur_node = allreduce_op_list[i];
        cur_node->add_control_dependency(pre_node);
      }
    }
  }
};

}  // namespace ngraph_bridge

}  // namespace tensorflow
