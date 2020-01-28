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
#ifndef NGRAPH_TF_NGRAPHVAR_H_
#define NGRAPH_TF_NGRAPHVAR_H_

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/default/logging.h"

#include "ngraph/event_tracing.hpp"
#include "ngraph/runtime/backend.hpp"

#include "ngraph_bridge/ngraph_backend_manager.h"
#include "ngraph_bridge/ngraph_freshness_tracker.h"
#include "ngraph_bridge/ngraph_utils.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

// THIS CLASS IS NOT BEING USED ANYWHERE
class NGraphVar : public ResourceBase {
 public:
  explicit NGraphVar(DataType dtype, TensorShape shape, string BackendName);

  // Not copyable or movable.
  NGraphVar(const NGraphVar&) = delete;
  NGraphVar& operator=(const NGraphVar&) = delete;

  mutex* mu() { return &mu_; }
  Tensor* tensor() { return &tf_tensor_; }
  shared_ptr<ngraph::runtime::Tensor> ng_tensor() { return ng_tensor_; };

  string DebugString() const override {
    return strings::StrCat(DataTypeString(tf_tensor_.dtype()), "/",
                           tf_tensor_.shape().DebugString());
  }

  // Copies the NG Tensor to TF Tensor for this variable
  // Involves a copy from device to host
  // Returns the number of tensor copies made (0 or 1)
  int copy_ng_to_tf();

  // Copies the TF Tensor to NG Tensor for this variable
  // Involves a copy from host to device
  // Returns the number of tensor copies made (0 or 1)
  int copy_tf_to_ng();

  // updates the NGTensor with the new value
  // This new_value could be from ngraph-tensor, for e.g. when computed from
  // NGraphEncapsulateOp
  // and saved in Catalog
  // Returns the number of tensor copies made (0 or 1)
  int update_ng_tensor(shared_ptr<ngraph::runtime::Tensor> new_value);

  // updates the NGTensor with the new value
  // This new_value could be from tf-tensor, for e.g. when computed from a TF op
  // Returns the number of tensor copies made (0 or 1)
  int update_ng_tensor(Tensor* new_value);

 private:
  mutex mu_;
  Tensor tf_tensor_;
  shared_ptr<ngraph::runtime::Tensor> ng_tensor_;
  string ng_backend_name_;
  bool ng_tf_share_buffer_;
  ~NGraphVar() override {
    // Release the backend
    NGRAPH_VLOG(2) << "~NGraphVar::ReleaseBackend";
    BackendManager::ReleaseBackend(ng_backend_name_);
  }
};

}  // namespace ng-bridge
}  // namespace tf

#endif
