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
#ifndef NGRAPH_TF_BRIDGE_BUILDER_H_
#define NGRAPH_TF_BRIDGE_BUILDER_H_

#include <ostream>
#include <vector>

#include "ngraph/ngraph.hpp"

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"

using namespace std;
namespace tf = tensorflow;
namespace ng = ngraph;

namespace ngraph_bridge {
class Builder {
 public:
  static tf::Status TranslateGraph(const std::vector<tf::TensorShape>& inputs,
                                   const tf::Graph* tf_graph,
                                   shared_ptr<ng::Function>& ng_function);

  using OpMap = unordered_map<string, std::vector<shared_ptr<ng::Node>>>;

 private:
};
}  // namespace ngraph_bridge

#endif
