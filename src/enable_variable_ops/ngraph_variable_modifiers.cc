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

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/strings/strcat.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/default/logging.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "ngraph/runtime/backend.hpp"
#include "ngraph_backend_manager.h"
#include "ngraph_catalog.h"
#include "ngraph_freshness_tracker.h"
#include "ngraph_timer.h"
#include "ngraph_utils.h"
#include "ngraph_var.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

/* -------------------------------------------------
//
// NGraphApplyGradientDescentOp
//
---------------------------------------------------*/

class NGraphApplyGradientDescentOp : public OpKernel {
 private:
 public:
  explicit NGraphApplyGradientDescentOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES(context, false,
                errors::Internal("This constructor should not get called",
                                 name(), "\n"));
  }

  //---------------------------------------------------------------------------
  //  ~NGraphApplyGradientDescentOp()
  //---------------------------------------------------------------------------
  ~NGraphApplyGradientDescentOp() override {}

  // This will never be called
  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(
        context, false,
        errors::Internal("This kernel should not get called", name(), "\n"));
  }  // end of compute function
};   // end of NGraphApplyGradientDescent class definition

REGISTER_OP("NGraphApplyGradientDescent")
    .Input("var: Ref(T)")
    .Input("alpha: T")
    .Input("delta: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("just_looking: bool = false")
    .Attr("copy_to_tf: bool = false")
    .Attr("ngraph_graph_id: int");

REGISTER_KERNEL_BUILDER(Name("NGraphApplyGradientDescent").Device(DEVICE_CPU),
                        NGraphApplyGradientDescentOp);

/* -------------------------------------------------
//
// NGraphAssignSubOp
//
---------------------------------------------------*/

// Computes *input[0] = *input[0] - input[1]
class NGraphAssignSubOp : public OpKernel {
 private:
  // bool use_exclusive_lock_; //TF op has this
  ~NGraphAssignSubOp() override {}

 public:
  explicit NGraphAssignSubOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES(context, false,
                errors::Internal("This constructor should not get called",
                                 name(), "\n"));
  }

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(
        context, false,
        errors::Internal("This kernel should not get called", name(), "\n"));
  }
};

REGISTER_OP("NGraphAssignSub")
    .Input("ref: Ref(T)")
    .Input("value: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: type")
    .Attr("validate_shape: bool = true")
    .Attr("use_locking: bool = true")
    .Attr("just_looking: bool = false")
    .Attr("copy_to_tf: bool = false")
    .Attr("ngraph_graph_id: int");

REGISTER_KERNEL_BUILDER(Name("NGraphAssignSub").Device(DEVICE_CPU),
                        NGraphAssignSubOp);

/* -------------------------------------------------
//
// NGraphAssignAddOp
//
---------------------------------------------------*/

// Computes *input[0] = *input[0] + input[1]
class NGraphAssignAddOp : public OpKernel {
 public:
  explicit NGraphAssignAddOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES(context, false,
                errors::Internal("This constructor should not get called",
                                 name(), "\n"));
  }

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(
        context, false,
        errors::Internal("This kernel should not get called", name(), "\n"));
  }

 private:
  ~NGraphAssignAddOp() override {}
};

REGISTER_OP("NGraphAssignAdd")
    .Input("ref: Ref(T)")
    .Input("value: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: type")
    .Attr("validate_shape: bool = true")
    .Attr("use_locking: bool = true")
    .Attr("just_looking: bool = false")
    .Attr("copy_to_tf: bool = false")
    .Attr("ngraph_graph_id: int");

REGISTER_KERNEL_BUILDER(Name("NGraphAssignAdd").Device(DEVICE_CPU),
                        NGraphAssignAddOp);

}  // namespace ngraph_bridge

}  // namespace tensorflow
