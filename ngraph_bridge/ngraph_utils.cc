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

#if defined NGRAPH_DISTRIBUTED
#include "ngraph/distributed.hpp"
#endif

#include "ngraph_bridge/ngraph_utils.h"
#include "ngraph_bridge/version.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

Status IsNgraphTFLogTensorCopiesEnabled(int graph_id,
                                        bool& is_copy_log_enabled) {
  const char* copy_env_var = std::getenv("NGRAPH_TF_LOG_TENSOR_COPIES");
  if (copy_env_var == nullptr) {
    is_copy_log_enabled = false;
    return Status::OK();
  }
  int test_graph_id;
  try {
    test_graph_id = stoi(string(copy_env_var));
  } catch (const std::invalid_argument& ia) {
    return errors::InvalidArgument(
        "Invalid argument for NGRAPH_TF_LOG_TENSOR_COPIES. Exception: ",
        ia.what());
  }
  // if -1 copies are logged for all graphs
  is_copy_log_enabled = (test_graph_id == -1 || test_graph_id == graph_id);
  return Status::OK();
}

Status GetNgraphVarBufferSharingState(int& buffer_sharing_state) {
  const char* ngvar_buffer_env_var =
      std::getenv("NGRAPH_TF_NGVARIABLE_BUFFER_SHARING");
  if (ngvar_buffer_env_var == nullptr) {
    buffer_sharing_state = -1;
    return Status::OK();
  }
  int env_var_val;
  try {
    env_var_val = stoi(string(ngvar_buffer_env_var));
  } catch (const std::invalid_argument& ia) {
    return errors::InvalidArgument(
        "Invalid argument for NGRAPH_TF_NGVARIABLE_BUFFER_SHARING. Exception: ",
        ia.what());
  }
  if (env_var_val != 0 && env_var_val != 1) {
    return errors::InvalidArgument(
        "Invalid argument for NGRAPH_TF_NGVARIABLE_BUFFER_SHARING. Pass 1 to "
        "enable, 0 to disable");
  }

  buffer_sharing_state = env_var_val;
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
  if (ngraph_tf_are_variables_enabled())
    return (node_type == "NGraphVariable" || node_type == "NGraphAssign");
  else
    return node_type == "NGraphVariable";
}

bool IsNGSupportedType(string node_type) {
  return (IsNGVariableType(node_type) || node_type == "NGraphEncapsulate");
};

// Read from this ng_tensor into tf_tensor
void ReadNGTensor(shared_ptr<ng::runtime::Tensor> ng_tensor,
                  Tensor* tf_tensor) {
  ngraph::Event event_sync_ng_tf_tensors("Tensor Read D2H", "", "");
  void* tf_src_ptr = (void*)DMAHelper::base(tf_tensor);
  ng_tensor->read(tf_src_ptr, 0, ng_tensor->get_element_count() *
                                     ng_tensor->get_element_type().size());
  event_sync_ng_tf_tensors.Stop();
  ngraph::Event::write_trace(event_sync_ng_tf_tensors);
}

// Write into this ng_tensor from tf_tensor
void WriteNGTensor(shared_ptr<ng::runtime::Tensor> ng_tensor,
                   Tensor* tf_tensor) {
  ngraph::Event event_sync_ng_tf_tensors("Tensor Write H2D", "", "");
  void* tf_src_ptr = (void*)DMAHelper::base(tf_tensor);
  ng_tensor->write(tf_src_ptr, 0, ng_tensor->get_element_count() *
                                      ng_tensor->get_element_type().size());
  event_sync_ng_tf_tensors.Stop();
  ngraph::Event::write_trace(event_sync_ng_tf_tensors);
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

//---------------------------------------------------------------------------
//  TensorToStream
//---------------------------------------------------------------------------
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
    case DataType::DT_INT8:
      *ng_et = ng::element::i8;
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

Status NgraphSerialize(const std::string& file_name,
                       const std::shared_ptr<ngraph::Function>& ng_function) {
  int json_indentation = 4;
  string serialized;
  if (ng_function == nullptr) {
    return errors::Internal(
        "Passed a null pointer as ng function to serialize");
  }
  try {
    ngraph::serialize(ng_function, json_indentation);
  } catch (...) {
    return errors::Internal("Failed to serialize ngraph function");
  }
  return StringToFile(file_name, serialized, true);
}

string SanitizeFileName(const string file_name) {
  // Sanitizing file name to take care of '/' that might be present in TF node
  // names
  string new_file_name;
  // The valid TF node names seem to be: [A-Za-z0-9.][A-Za-z0-9_.\\-/]*
  // . is another non-alphanumeric char, but once / are replaced by --, . is
  // fine in a file name.
  for (const auto& itr : file_name) {
    new_file_name += ((itr == '/') ? string("--") : string({itr}));
  }
  return new_file_name;
}

Status StringToFile(const std::string& file_name, const std::string& contents,
                    bool sanitize_name) {
  string new_file_name =
      sanitize_name ? SanitizeFileName(file_name) : file_name;
  NGRAPH_VLOG(0) << "Serializing graph to: " << new_file_name << std::endl;
  std::ofstream f;
  f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  try {
    f.open(new_file_name);
    f << contents;
    f.close();
  } catch (std::ofstream::failure& e) {
    NGRAPH_VLOG(0) << "Exception opening/closing file " << new_file_name
                   << std::endl;
    NGRAPH_VLOG(0) << e.what() << std::endl;
    return errors::Internal("Failed to dump string to file. Filename: ",
                            new_file_name, ". Exception: ", e.what());
  }
  return Status::OK();
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
  int rank_id = ng::get_distributed_interface()->get_rank();
  ss << "_" << std::setfill('0') << std::setw(4) << rank_id;
#endif
  return ss.str();
}

std::string GraphFilenamePrefix(std::string kind, int idx, int sub_idx) {
  std::stringstream ss;
  ss << GraphFilenamePrefix(kind, idx) << "_" << std::setfill('0')
     << std::setw(4) << sub_idx;
#if defined NGRAPH_DISTRIBUTED
  int rank_id = ng::get_distributed_interface()->get_rank();
  ss << "_" << std::setfill('0') << std::setw(4) << rank_id;
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

#if defined(NGRAPH_DISTRIBUTED)
void OpControlOrder(const std::shared_ptr<ngraph::Function>& ng_function,
                    const std::string& op_name) {
  // Get the serialized ops and stored the allreduce ops to a vector and
  ng::NodeVector op_list;
  for (const shared_ptr<ng::Node>& node : ng_function->get_ordered_ops()) {
    if (node->description() == op_name) {
      op_list.push_back(node);
    }
  }
  // Sort the allreduce ops according to the TF names
  std::sort(op_list.begin(), op_list.end(),
            [](const shared_ptr<ng::Node>& x, const shared_ptr<ng::Node>& y) {
              return x->get_friendly_name() < y->get_friendly_name();
            });
  // Add control dependency in for the allreduce ops
  if (op_list.size() > 1) {
    for (size_t i = 1; i < op_list.size(); ++i) {
      auto pre_node = op_list[i - 1];
      auto cur_node = op_list[i];
      cur_node->add_control_dependency(pre_node);
    }
  }
}
#endif

bool IsProcessedByNgraphPass(Graph* g) {
  // TODO: place a dummy node as a marker
  // Current method may fail when graph has no encapsulates after first pass
  // Also variable/optimizer change introduces other types of ng nodes
  for (Node* node : g->nodes()) {
    if (node->type_string() == "NGraphEncapsulate") return true;
  }
  return false;
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
