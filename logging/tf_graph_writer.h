/*******************************************************************************
 * Copyright (C) 2021-2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/
#ifndef TF_GRAPH_WRITER_H_
#define TF_GRAPH_WRITER_H_

#include <ostream>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

namespace tensorflow {

namespace openvino_tensorflow {

// GraphToDot
// Transforms a TensorFlow graph to a DOT file for rendering with graphviz
std::string GraphToDot(Graph* graph, const std::string& title);

// GraphToDotFile
// Saves a TensorFlow graph into a DOT file for rendering with graphviz
void GraphToDotFile(Graph* graph, const std::string& filename,
                    const std::string& title);

// GraphToPbTextFile
// Saves a TensorFlow graph into a protobuf text
void GraphToPbTextFile(Graph* graph, const std::string& filename);

// GraphToPbFile
// Saves a TensorFlow graph into a protobuf
void GraphToPbFile(Graph* graph, const std::string& filename);

// PbTextFileToDotFile
// Saves a protobuf text into a DOT file
void PbTextFileToDotFile(const std::string& pbtxt_filename,
                         const std::string& dot_filename,
                         const std::string& title);

}  // namespace openvino_tensorflow

}  // namespace tensorflow

#endif  // TF_GRAPH_WRITER_H_
