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

//
// Derived from tensorflow/tensorflow/core/kernels/variable_ops.h.
//
#ifndef NGRAPH_TF_VARIABLE_OPS_H_
#define NGRAPH_TF_VARIABLE_OPS_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

#include "ngraph/ngraph.hpp"

namespace ng = ngraph;
namespace tf = tensorflow;

namespace tensorflow {

class NGraphVariableOp : public OpKernel {
 public:
  explicit NGraphVariableOp(OpKernelConstruction* context);
  void Compute(OpKernelContext* ctx) override;

 private:
  DataType dtype_;
  TensorShape shape_;

  tf::mutex init_mu_;
  ContainerInfo cinfo_ GUARDED_BY(init_mu_);
  bool initialized_ GUARDED_BY(init_mu_){false};

  TF_DISALLOW_COPY_AND_ASSIGN(NGraphVariableOp);
};

} // namespace tensorflow

#endif // NGRAPH_TF_VARIABLE_OPS_H_
