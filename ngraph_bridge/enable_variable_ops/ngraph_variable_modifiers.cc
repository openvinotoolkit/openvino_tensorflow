/*******************************************************************************
 * Copyright 2019-2020 Intel Corporation
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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/default/logging.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "ngraph/runtime/backend.hpp"

#include "ngraph_bridge/enable_variable_ops/ngraph_var.h"
#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/ngraph_catalog.h"
#include "ngraph_bridge/ngraph_timer.h"
#include "ngraph_bridge/ngraph_utils.h"

#include "ngraph_bridge/ngraph_register_stub_kernels.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

// Register NGraphOptimizers here
// These Optimizer Ops are replaced by a TF computational subgraph
// in ReplaceModifiers Rewrite Pass. Hence, these Stub Kernels/Op will never get
// called

// Keep them in alphabetical order
REGISTER_NGRAPH_STUB_KERNEL("NGraphApplyGradientDescent");
REGISTER_NGRAPH_STUB_KERNEL("NGraphApplyMomentum");
REGISTER_NGRAPH_STUB_KERNEL(
    "NGraphAssignAdd");  //*input[0] = *input[0] + input[1]
REGISTER_NGRAPH_STUB_KERNEL(
    "NGraphAssignSub");  //*input[0] = *input[0] - input[1]

}  // namespace ngraph_bridge

}  // namespace tensorflow
