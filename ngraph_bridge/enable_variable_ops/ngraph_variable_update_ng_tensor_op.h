/*******************************************************************************
 * Copyright 2019 Intel Corporation
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

#ifndef NGRAPH_TF_VARIABLE_UPDATE_NG_TENSOR_OP_H_
#define NGRAPH_TF_VARIABLE_UPDATE_NG_TENSOR_OP_H_
#pragma once

#include <ostream>

#include "tensorflow/core/graph/graph.h"

/* -------------------------------------------------
//
// NGraphVariableUpdateNGTensor
//
---------------------------------------------------*/

namespace tensorflow {

namespace ngraph_bridge {

class NGraphVariableUpdateNGTensorOp : public OpKernel {
 public:
  explicit NGraphVariableUpdateNGTensorOp(OpKernelConstruction* ctx);
  ~NGraphVariableUpdateNGTensorOp() override;
  void Compute(OpKernelContext* ctx) override;

 private:
  int ng_graph_id_;
  string ng_variable_shared_name_;
};

}  // namespace ngraph_bridge

}  // namespace tensorflow
#endif  // NGRAPH_TF_VARIABLE_UPDATE_NG_TENSOR_OP_H_
