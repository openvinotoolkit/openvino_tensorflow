/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
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

#include "test/opexecuter.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

// Test op: CropAndResize
// Disabled till a backend starts supporting it
// Add more tests when backend starts supporting

TEST(ImageOps, DISABLED_CropAndResize) {
  Scope root = Scope::NewRootScope();

  // TODO check if assigning random values is ok,
  // or we should probably assign some known in-range values
  // Fix and enable this test when a backend supports CropAndResize

  // [batch, height, width, channels]
  Tensor image(DT_FLOAT, TensorShape({4, 64, 64, 3}));
  AssignInputValuesRandom(image);

  int num_boxes = 5;
  // [num_boxes, 4]
  Tensor boxes(DT_FLOAT, TensorShape({num_boxes, 4}));
  AssignInputValuesRandom(boxes);

  // [num_boxes]
  Tensor box_ind(DT_INT32, TensorShape({num_boxes}));
  AssignInputValuesRandom(box_ind);

  // [crop_height, crop_width]
  Tensor crop_size(DT_INT32, TensorShape({6, 5}));
  AssignInputValuesRandom(crop_size);

  // TODO check with non-zero extrapolation value and method="nearest"
  auto attr =
      ops::CropAndResize::Attrs().ExtrapolationValue(0.0).Method("bilinear");

  auto R = ops::CropAndResize(root, image, boxes, box_ind, crop_size, attr);
  OpExecuter opexecuter(root, "CropAndResize", {R});
  opexecuter.RunTest();
}

}  // testing
}  // ngraph-bridge
}  // tensorflow
