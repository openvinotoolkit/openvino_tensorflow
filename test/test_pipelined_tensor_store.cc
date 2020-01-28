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

#include "gtest/gtest.h"

#include "tensorflow/core/common_runtime/dma_helper.h"

#include "ngraph_bridge/ngraph_pipelined_tensors.h"
#include "test/test_utilities.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

TEST(PipelinedTensorStoreTest, SimpleTest) {
  int in_depth = 2;
  int out_depth = 3;
  PipelinedTensorMatrix pipelined_input_tensors(in_depth);
  PipelinedTensorMatrix pipelined_output_tensors(out_depth);
  ASSERT_THROW(new PipelinedTensorsStore(pipelined_input_tensors,
                                         pipelined_output_tensors),
               std::runtime_error);
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow