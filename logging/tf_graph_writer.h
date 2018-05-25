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
#ifndef TF_GRAPH_WRITER_H_
#define TF_GRAPH_WRITER_H_

#include <ostream>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

using namespace std;
namespace tf = tensorflow;
namespace ngraph_bridge {

// GraphToDot
// Transforms a TensorFlow graph to a DOT file for rendering with graphviz
std::string GraphToDot(tf::Graph* graph, const std::string& title,
                       bool annotate_device);

// GraphToDotFile
// Saves a TensorFlow graph into a DOT file for rendering with graphviz
void GraphToDotFile(tf::Graph* graph, const std::string& filename,
                    const std::string& title, bool annotate_device);

// GraphToPbTextFile
// Saves a TensorFlow graph into a protobuf text
void GraphToPbTextFile(tf::Graph* graph, const string& filename);
}  // namespace ngraph_bridge

#endif  // TF_GRAPH_WRITER_H_
