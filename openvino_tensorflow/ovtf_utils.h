/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#pragma once

#ifndef OPENVINO_TF_BRIDGE_UTILS_H_
#define OPENVINO_TF_BRIDGE_UTILS_H_

#include <fstream>
#include <ostream>
#include <sstream>

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

#include "ngraph/ngraph.hpp"

#include "logging/ovtf_log.h"
#include "logging/tf_graph_writer.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {
namespace util {

bool IsAlreadyProcessed(Graph* g);

// Descending sort the map based on the value
void PrintNodeHistogram(const std::unordered_map<string, int>&,
                        bool sorted = true);

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

// Collect the total memory usage through /proc/self/stat
void MemoryProfile(long&, long&);

// Check if we're supposed to dump graphs
bool DumpAllGraphs();
// Dump TF graphs in .pbtxt format
void DumpTFGraph(tensorflow::Graph* graph, int idx, string filename_prefix);
// Dump nGraph graphs in .dot format
void DumpNGGraph(std::shared_ptr<ngraph::Function> function,
                 const string filename_prefix);

// Get an environment variable
string GetEnv(const std::string& env);

// Get backend device names 
std::vector<std::string> GetDeviceIds(string device_name);

// Set the environment variable env with val
void SetEnv(const char* env, const char* val);

}  // namespace util
}  // namespace openvino_tensorflow
}  // namespace tensorflow

#endif  // OPENVINO_TF_BRIDGE_UTILS_H_
