/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
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

#include "ngraph_bridge/ngraph_utils.h"
#include "ngraph_bridge/version.h"

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {
namespace util {

template <typename T>
static void TensorDataToStream(std::ostream& ostream, int64 n_elements,
                               const char* data) {
  const T* data_T = reinterpret_cast<const T*>(data);
  for (size_t i = 0; i < n_elements; i++) {
    ostream << data_T[i] << ",";
  }
}

Status TensorToStream(std::ostream& ostream, const Tensor& tensor) {
  const char* data = tensor.tensor_data().data();
  int64 n_elements = tensor.NumElements();
  switch (tensor.dtype()) {
    case DT_HALF:
      TensorDataToStream<Eigen::half>(ostream, n_elements, data);
      break;
    case DT_FLOAT:
      TensorDataToStream<float>(ostream, n_elements, data);
      break;
    case DT_DOUBLE:
      TensorDataToStream<double>(ostream, n_elements, data);
      break;
    case DT_UINT32:
      TensorDataToStream<uint32>(ostream, n_elements, data);
      break;
    case DT_INT32:
      TensorDataToStream<int32>(ostream, n_elements, data);
      break;
    case DT_UINT8:
    case DT_QUINT8:
      TensorDataToStream<uint8>(ostream, n_elements, data);
      break;
    case DT_UINT16:
    case DT_QUINT16:
      TensorDataToStream<uint16>(ostream, n_elements, data);
      break;
    case DT_INT8:
    case DT_QINT8:
      TensorDataToStream<int8>(ostream, n_elements, data);
      break;
    case DT_INT16:
    case DT_QINT16:
      TensorDataToStream<int16>(ostream, n_elements, data);
      break;
    case DT_UINT64:
      TensorDataToStream<uint64>(ostream, n_elements, data);
      break;
    case DT_INT64:
      TensorDataToStream<int64>(ostream, n_elements, data);
      break;
    case DT_BOOL:
      TensorDataToStream<bool>(ostream, n_elements, data);
      break;
    case DT_BFLOAT16:
      return errors::Internal(
          "TensorToStream got data type bfloat16. No compatible standard C++ "
          "data type.");
      break;
    default:
      return errors::Internal("TensorToStream got unsupported data type ",
                              DataType_Name(tensor.dtype()));
      break;
  }
  return Status::OK();
}

Status TFDataTypeToNGraphElementType(DataType tf_dt,
                                     ngraph::element::Type* ng_et) {
  switch (tf_dt) {
    case DataType::DT_FLOAT:
      *ng_et = ngraph::element::f32;
      break;
    case DataType::DT_DOUBLE:
      *ng_et = ngraph::element::f64;
      break;
    case DataType::DT_INT32:
      *ng_et = ngraph::element::i32;
      break;
    case DataType::DT_UINT8:
      *ng_et = ngraph::element::u8;
      break;
    case DataType::DT_INT8:
      *ng_et = ngraph::element::i8;
      break;
    case DataType::DT_UINT16:
      *ng_et = ngraph::element::u16;
      break;
    case DataType::DT_INT64:
      *ng_et = ngraph::element::i64;
      break;
    case DataType::DT_UINT32:
      *ng_et = ngraph::element::u32;
      break;
    case DataType::DT_UINT64:
      *ng_et = ngraph::element::u64;
      break;
    case DataType::DT_BOOL:
      *ng_et = ngraph::element::boolean;
      break;
    case DataType::DT_QINT8:
      *ng_et = ngraph::element::i8;
      break;
    case DataType::DT_QUINT8:
      *ng_et = ngraph::element::u8;
      break;
    case DataType::DT_QINT32:
      *ng_et = ngraph::element::i32;
      break;
    case DataType::DT_BFLOAT16:
      *ng_et = ngraph::element::bf16;
      break;
    case DataType::DT_HALF:
      *ng_et = ngraph::element::f16;
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

void PrintNodeHistogram(const std::unordered_map<string, int>& histogram,
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
  std::cout << std::endl;
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
    vector<string> mem_str = ngraph::split(mem_in, ' ');
    vsize = std::stol(mem_str[22]);
    rss = std::stol(mem_str[23]);

    long page_size_kb = sysconf(_SC_PAGE_SIZE) /
                        1024;  // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024;   // unit kb
    resident_set = rss * page_size_kb;
  }
}

void DumpTFGraph(tensorflow::Graph* graph, int idx, std::string filename) {
  if (!DumpAllGraphs()) {
    return;
  }

  std::stringstream ss;
  ss << filename << "_" << std::setfill('0') << std::setw(4) << idx;
  NGRAPH_VLOG(0) << "Dumping TF graph to " << ss.str() + ".pbtxt";
  GraphToPbTextFile(graph, ss.str() + ".pbtxt");
}

void DumpNGGraph(std::shared_ptr<ngraph::Function> function,
                 const string filename) {
  if (!DumpAllGraphs()) {
    return;
  }

  NGRAPH_VLOG(0) << "Dumping nGraph graph to " << filename + ".dot";
  // enable shape info for nGraph graphs
  SetEnv("NGRAPH_VISUALIZE_TREE_OUTPUT_SHAPES", "1");
  SetEnv("NGRAPH_VISUALIZE_TREE_OUTPUT_TYPES", "1");
  SetEnv("NGRAPH_VISUALIZE_TREE_IO", "1");
  ngraph::plot_graph(function, filename + ".dot");
}

bool DumpAllGraphs() { return GetEnv("NGRAPH_TF_DUMP_GRAPHS") == "1"; }

bool IsAlreadyProcessed(Graph* g) {
  // TODO: place a dummy node as a marker
  // Current method may fail when graph has no encapsulates after first pass
  for (Node* node : g->nodes()) {
    if (node->type_string() == "_nGraphEncapsulate") return true;
  }
  return false;
}

string GetEnv(const char* env) {
  const char* val = std::getenv(env);
  if (val == nullptr) {
    return "";
  } else {
    return string(val);
  }
}

void SetEnv(const char* env, const char* val) { setenv(env, val, 1); }

}  // namespace util
}  // namespace ngraph_bridge
}  // namespace tensorflow
