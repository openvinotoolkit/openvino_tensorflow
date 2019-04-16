/*******************************************************************************
 * Copyright 2017-2019 Intel Corporation
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

#include "test_utilities.h"
#include <assert.h>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include "ngraph_log.h"

using namespace std;

namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

void ActivateNGraph() {
  setenv("NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS", "1", 1);
  unsetenv("NGRAPH_TF_DISABLE");
}

void DeactivateNGraph() {
  unsetenv("NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS");
  setenv("NGRAPH_TF_DISABLE", "1", 1);
}

// Input x will be used as an anchor
// Actual value assigned equals to x * i
void AssignInputValuesAnchor(Tensor& A, float x) {
  auto A_flat = A.flat<float>();
  auto A_flat_data = A_flat.data();
  for (int i = 0; i < A_flat.size(); i++) {
    A_flat_data[i] = x * i;
  }
}

// Randomly generate a float number between -10.00 ~ 10.99
void AssignInputValuesRandom(Tensor& A) {
  auto A_flat = A.flat<float>();
  auto A_flat_data = A_flat.data();
  srand(static_cast<unsigned>(time(0)));
  for (int i = 0; i < A_flat.size(); i++) {
    // give a number between 0 and 20
    float value =
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX / 20.0f);
    value = (value - 10.0f);              // range from -10 to 10
    value = roundf(value * 100) / 100.0;  // change the precision of the float
                                          // to 2 number after the decimal
    A_flat_data[i] = value;
  }
}

void PrintTensor(const Tensor& T1) {
  LOG(INFO) << "print tensor values" << T1.DebugString();
}

// Only displays values in tensor without shape information etc.
void PrintTensorAllValues(const Tensor& T1, int64 max_entries) {
  LOG(INFO) << "all tensor values" << T1.SummarizeValue(max_entries) << endl;
}

// Compares Tensors considering tolerance
void Compare(Tensor& T1, Tensor& T2, float tol) {
  // Assert rank
  ASSERT_EQ(T1.dims(), T2.dims())
      << "Ranks unequal for T1 and T2. T1.shape = " << T1.shape()
      << " T2.shape = " << T2.shape();

  // Assert each dimension
  for (int i = 0; i < T1.dims(); i++) {
    ASSERT_EQ(T1.dim_size(i), T2.dim_size(i))
        << "T1 and T2 shapes do not match in dimension " << i
        << ". T1.shape = " << T1.shape() << " T2.shape = " << T2.shape();
  }

  // Assert type
  ASSERT_EQ(T1.dtype(), T2.dtype()) << "Types of T1 and T2 did not match";

  auto T_size = T1.flat<float>().size();
  auto T1_data = T1.flat<float>().data();
  auto T2_data = T2.flat<float>().data();
  for (int k = 0; k < T_size; k++) {
    auto a = T1_data[k];
    auto b = T2_data[k];
    if (a == 0) {
      EXPECT_NEAR(a, b, tol);
    } else {
      auto rel = a - b;
      auto rel_div = std::abs(rel / a);
      EXPECT_TRUE(rel_div < tol);
    }
  }
}

// Compares Tensor vectors
void Compare(const vector<Tensor>& v1, const vector<Tensor>& v2, float rtol,
             float atol) {
  ASSERT_EQ(v1.size(), v2.size()) << "Length of 2 tensor vectors do not match.";
  for (int i = 0; i < v1.size(); i++) {
    NGRAPH_VLOG(3) << "Comparing output at index " << i;
    auto expected_dtype = v1[i].dtype();
    switch (expected_dtype) {
      case DT_FLOAT:
        Compare<float>(v1[i], v2[i], rtol, atol);
        break;
      case DT_INT8:
        Compare<int8>(v1[i], v2[i]);
        break;
      case DT_INT16:
        Compare<int16>(v1[i], v2[i]);
        break;
      case DT_INT32:
        Compare<int>(v1[i], v2[i]);
        break;
      case DT_INT64:
        Compare<int64>(v1[i], v2[i]);
        break;
      case DT_BOOL:
        Compare<bool>(v1[i], v2[i]);
        break;
      case DT_QINT8:
        Compare<qint8>(v1[i], v2[i]);
        break;
      case DT_QUINT8:
        Compare<quint8>(v1[i], v2[i]);
        break;
      default:
        ASSERT_TRUE(false)
            << "Could not find the corresponding function for the "
               "expected output datatype."
            << expected_dtype;
    }
  }
}

// Specialized template for Comparing float
template <>
bool Compare(float desired, float actual, float rtol, float atol) {
  if (desired == 0 && actual == 0) {
    return true;
  } else {
    // same as numpy.testing.assert_allclose
    return std::abs(desired - actual) <= (atol + rtol * std::abs(desired));
  }
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
