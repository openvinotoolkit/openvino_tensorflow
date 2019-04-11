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
#ifndef NGRAPH_TF_NgraphVar_H_
#define NGRAPH_TF_NgraphVar_H_

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/strings/strcat.h"

#include "tensorflow/core/platform/default/logging.h"

#include "ngraph/runtime/backend.hpp"
#include "ngraph_backend_manager.h"
#include "ngraph_freshness_tracker.h"
#include "ngraph_utils.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

// THIS CLASS IS NOT BEING USED ANYWHERE
class NGraphVar : public ResourceBase {
 public:
  explicit NGraphVar(DataType dtype, TensorShape shape, string BackendName)
      : tf_tensor_(dtype, shape),
        ng_backend_name_(BackendName),
        sync_ng_tensor_(false) {
    // TF datatype to nGraph element type
    ng::element::Type ng_element_type;
    TFDataTypeToNGraphElementType(dtype, &ng_element_type);

    // TF TensorShape to nGraphShape
    ng::Shape ng_shape(shape.dims());
    for (int j = 0; j < shape.dims(); ++j) {
      ng_shape[j] = shape.dim_size(j);
    }

    // Create Backend
    NGRAPH_VLOG(4) << "NGraphVar::Create Backend ";
    BackendManager::CreateBackend(ng_backend_name_);
    ng::runtime::Backend* op_backend =
        BackendManager::GetBackend(ng_backend_name_);

    // Create nGTensor
    ng_tensor_ = op_backend->create_tensor(ng_element_type, ng_shape);
  }
  // Not copyable or movable.
  NGraphVar(const NGraphVar&) = delete;
  NGraphVar& operator=(const NGraphVar&) = delete;

  mutex* mu() { return &mu_; }
  Tensor* tensor() { return &tf_tensor_; }
  shared_ptr<ngraph::runtime::Tensor> ng_tensor() { return ng_tensor_; };

  string DebugString() override {
    return strings::StrCat(DataTypeString(tf_tensor_.dtype()), "/",
                           tf_tensor_.shape().DebugString());
  }

  bool need_sync_ng_tensor() { return sync_ng_tensor_; }
  void sync_ng_tensor(bool sync_ng_tensor) { sync_ng_tensor_ = sync_ng_tensor; }

  // TODO(malikshr): Implement syncing utility functions here
  Status copy_ng_to_tf();
  Status copy_tf_to_ng();

 private:
  mutex mu_;
  Tensor tf_tensor_;
  shared_ptr<ngraph::runtime::Tensor> ng_tensor_;
  string ng_backend_name_;
  // sync from tf to ng
  bool sync_ng_tensor_;
  ~NGraphVar() override {
    // Release the backend
    NGRAPH_VLOG(2) << "~NGraphVar::ReleaseBackend";
    BackendManager::ReleaseBackend(ng_backend_name_);
  }
};

}  // namespace ng-bridge
}  // namespace tf

#endif
